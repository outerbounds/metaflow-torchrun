from metaflow.plugins.parallel_decorator import (
    ParallelDecorator,
    _local_multinode_control_task_step_func,
    UBF_CONTROL,
)
from metaflow.exception import MetaflowException
from metaflow import current
from functools import partial
import subprocess
import socket
import time
import json
import sys
import os

NODE_STARTED_VAR = "torchrun_node_started"


class TorchRunExecutor:
    def __init__(
        self, pathspec, main_addr, main_port, num_nodes, node_index, nproc_per_node=1
    ) -> None:
        self.torchrun_args = {
            "rdzv-id": "123",
            "rdzv_endpoint": "%s:%s" % (main_addr, main_port),
            "nnodes": num_nodes,
            "master_addr": main_addr,
            "master_port": main_port,
            "node_rank": node_index,
            "rdzv-backend": "c10d",
            "max-restarts": 3,
        }
        self.nproc_per_node = nproc_per_node

    def run(
        self,
        entrypoint,
        entrypoint_args=None,
        entrypoint_args_raw=None,
        nproc_per_node=None,
    ):
        """
        User-facing function that calls the torchrun command.
        `entry_point_args` : Dict | None
        `entrypoint_args_raw` : List[str] | None
            Either `entry_point_args` or `entrypoint_args_raw` must be provided. Both cannot be provided.
        """
        if entrypoint_args is not None and entrypoint_args_raw is not None:
            raise ValueError(
                "Only one of `entry_point_args` or `entrypoint_args_raw` can be provided."
            )

        self._ensure_torch_installed()
        cmd = ["torchrun"]

        for arg, val in dict(
            **self.torchrun_args, nproc_per_node=nproc_per_node or self.nproc_per_node
        ).items():
            cmd.extend(["--%s" % arg, str(val)])
        cmd.append(entrypoint)

        if entrypoint_args is not None:
            for arg, val in entrypoint_args.items():
                cmd.extend(["--%s" % arg, str(val)])
        elif entrypoint_args_raw is not None:
            cmd.extend(entrypoint_args_raw)

        # logging.info("[IP - %s] %s" % (socket.gethostbyname(socket.gethostname()), " ".join(cmd)))

        try:
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            ) as process:
                while process.poll() is None:
                    stdout = process.stdout.read1()
                    try:
                        text = stdout.decode("utf-8")
                    except UnicodeDecodeError:
                        text = ""
                    print(text, end="", flush=True)

        except subprocess.CalledProcessError as e:
            print(e.stdout)
            raise e

    def _ensure_torch_installed(self):
        try:
            import torch
        except ImportError:
            raise MetaflowException(
                "PyTorch is not installed. Please install PyTorch before using the torchrun decorator."
            )


class TorchrunDecoratorParallel(ParallelDecorator):
    name = "torchrun"
    defaults = {
        "master_port": "3339",
        "all_nodes_started_timeout": 60 * 8,  # 8 minutes
        "nproc_per_node": 1,
    }
    IS_PARALLEL = True

    def _setup_current(self, main_addr, main_port, ubf_context, num_nodes, node_index):
        from metaflow import current

        # main_addr = current.parallel.main_ip
        # num_nodes = current.parallel.num_nodes
        # node_index = current.parallel.node_index

        torch_executor = TorchRunExecutor(
            pathspec=current.pathspec,
            main_addr=main_addr,
            main_port=main_port,
            num_nodes=num_nodes,
            node_index=node_index,
            nproc_per_node=self.nproc_per_node,
        )
        current._update_env({"torch": torch_executor})

    def step_init(self, flow, graph, step, decos, environment, flow_datastore, logger):
        from metaflow.plugins.aws.aws_utils import compute_resource_attributes

        for deco in decos:
            if deco.name in ["resources", "kubernetes", "batch"]:
                compute_deco_attrs = compute_resource_attributes(
                    decos, deco, {"cpu": "1", "gpu": "0"}
                )
                try:
                    self.nproc_per_node = int(compute_deco_attrs["gpu"])
                except KeyError:
                    self.nproc_per_node = int(compute_deco_attrs["cpu"])
                if not self.nproc_per_node > 0:
                    self.nproc_per_node = int(compute_deco_attrs["cpu"])
                break

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        def _step_func_with_setup():
            self.setup_distributed_env(flow, ubf_context)
            step_func()

        if (
            ubf_context == UBF_CONTROL
            and os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local"
        ):
            from functools import partial

            env_to_use = getattr(self.environment, "base_env", self.environment)

            return partial(
                _local_multinode_control_task_step_func,
                flow,
                env_to_use,
                _step_func_with_setup,
                retry_count,
            )
        else:
            return _step_func_with_setup

    def setup_distributed_env(self, run, ubf_context):
        def _num_nodes_started(path=NODE_STARTED_VAR):
            "Process run on control job to check if all nodes have sent alert they are started."
            objs = s3.get_recursive([path])
            num_started = 0
            for obj in objs:
                obj = json.loads(obj.text)
                if obj["node_started"]:
                    num_started += 1
                else:
                    raise WorkerFailedStartException(node_index)
            return num_started

        from metaflow import S3

        s3 = S3(run=run)

        # gather my node info
        if "AWS_BATCH_JOB_ID" in os.environ:
            num_nodes = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
            node_index = os.environ["AWS_BATCH_JOB_NODE_INDEX"]
            if ubf_context != UBF_CONTROL:
                main_addr = os.environ["AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS"]
            else:
                main_addr = socket.gethostbyname(socket.gethostname())
        else:  # kubernetes
            num_nodes = int(os.environ["WORLD_SIZE"])
            node_index = int(os.environ["RANK"])
            if ubf_context != UBF_CONTROL:
                node_index += 1  # artifact of kubernetes jobset in experimental Kubernetes parallel implementation. TBD.
                main_addr = os.environ["MASTER_ADDR"]
            else:
                main_addr = socket.gethostbyname(socket.gethostname())
        node_key = os.path.join(NODE_STARTED_VAR, "node_%s.json" % node_index)

        self._setup_current(
            main_addr,
            self.attributes["master_port"],
            ubf_context,
            num_nodes,
            node_index,
        )

        # alert community about my node info
        s3.put(node_key, json.dumps({"node_started": True}))

        if ubf_context == UBF_CONTROL:
            t0 = time.time()
            while _num_nodes_started() < num_nodes:
                if self.attributes["all_nodes_started_timeout"] <= time.time() - t0:
                    raise AllNodesStartupTimeoutException()
                time.sleep(10)

        s3.close()


def get_backend():
    try:
        import torch

        return torch.distributed.get_backend()
    except ImportError:
        return None


class AllNodesStartupTimeoutException(MetaflowException):
    headline = "All workers did not join cluster error"

    def __init__(self):
        msg = "Exiting job due to time out waiting for all workers to join cluster. You can set the timeout in @torchrun(all_nodes_started_timeout=X)"
        super(AllNodesStartupTimeoutException, self).__init__(msg)
