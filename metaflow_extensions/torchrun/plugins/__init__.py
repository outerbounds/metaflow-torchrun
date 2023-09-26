from metaflow.plugins.parallel_decorator import ParallelDecorator, _local_multinode_control_task_step_func, UBF_CONTROL
from metaflow.exception import MetaflowException
from metaflow import current
from functools import partial
import subprocess
import socket
import sys
import os


NODE_STARTED_VAR = "torchrun_node_started"

class TorchRunExecutor:

    def __init__(self, 
                 pathspec,
                 main_addr, 
                 main_port, 
                 num_nodes, 
                 node_index, 
                 ) -> None:
        
        self.torchrun_args = {
            "rdzv-id": "123",
            "rdzv_endpoint": "%s:%s" % (main_addr, main_port),
            "nnodes": num_nodes, 
            "master_addr": main_addr,
            "master_port": main_port, # batch doesn't matter. k8s depends on port opened in @kubernetes jobset. 
            "node_rank": node_index,
            "rdzv-backend": "c10d",
            "max-restarts": 3
        }
    
    def run(self, entrypoint,  entrypoint_args=None, entrypoint_args_raw=None, nproc_per_node=1):
        """
        `entry_point_args` : Dict | None
        `entrypoint_args_raw` : List[str] | None
            Either `entry_point_args` or `entrypoint_args_raw` must be provided. Both cannot be provided.
        """
        if entrypoint_args is not None and entrypoint_args_raw is not None:
            raise ValueError("Only one of `entry_point_args` or `entrypoint_args_raw` can be provided.")

        self._ensure_torch_installed()
        cmd = ["torchrun"] 
        
        for arg, val in dict(**self.torchrun_args, nproc_per_node=nproc_per_node).items():
            cmd.extend(["--%s" % arg, str(val)])
        cmd.append(entrypoint)
        
        if entrypoint_args is not None:
            for arg, val in entrypoint_args.items():
                cmd.extend(["--%s" % arg, str(val)])
        elif entrypoint_args_raw is not None:
            cmd.extend(entrypoint_args_raw)
        
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    
    def _ensure_torch_installed(self):
        try:
            import torch
        except ImportError:
            raise MetaflowException("PyTorch is not installed. Please install PyTorch before using the torchrun_parallel decorator.")


class TorchrunDecoratorParallel(ParallelDecorator):

    name = "torchrun_parallel"
    defaults = {
        "master_port": "3339",
    }
    IS_PARALLEL = True

    def _setup_current(self, main_port,  ubf_context):

        from metaflow import current

        main_addr = current.parallel.main_ip
        num_nodes = current.parallel.num_nodes
        node_index = current.parallel.node_index

        # TODO : Fixme: find a way these come only from `current` to ensure implementation hygiene.
        if not os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local":
            # collect variables from environment
            if "AWS_BATCH_JOB_ID" in os.environ:
                num_nodes = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
                node_index = int(os.environ["AWS_BATCH_JOB_NODE_INDEX"])
                if ubf_context == UBF_CONTROL:
                    main_addr = socket.gethostname()
                else:
                    main_ip = os.environ["AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS"]
                    main_addr = socket.gethostbyaddr(main_ip)[0]
            else: # kubernetes
                num_nodes = int(os.environ["WORLD_SIZE"])
                node_index = int(os.environ["RANK"])
                main_addr = os.environ["MASTER_ADDR"]
        
        torch_executor = TorchRunExecutor(
            pathspec=current.pathspec,
            main_addr=main_addr,
            main_port=main_port,
            num_nodes=num_nodes,
            node_index=node_index, 
        )
        current._update_env({
            "torch" : torch_executor
        })
    
    def task_pre_step(self, step_name, task_datastore, metadata, run_id, task_id, flow, graph, retry_count, max_user_code_retries, ubf_context, inputs):
        self._setup_current(self.attributes["master_port"], ubf_context)
 
def get_backend():
    try:
        import torch
        return torch.distributed.get_backend()
    except ImportError:
        return None

STEP_DECORATORS_DESC = [("torchrun_parallel", ".TorchrunDecoratorParallel")]
