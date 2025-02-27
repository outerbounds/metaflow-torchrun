from typing import List, Dict, Union
import json
import os
import sys
import time
import signal
from threading import Thread

import subprocess
from .exceptions import TorchrunException, TorchNotInstalledException
from .datastore import TorchrunDatastore
from .status_notifier import (
    TaskStatusNotifier,
    wait_for_task_completion,
    TaskFailedException,
)

from metaflow.exception import MetaflowException


def _dict_to_args(d: Dict[str, str]) -> List[str]:
    data = []

    for k, v in d.items():
        if (v == "" or v is None):
            data.append(f"--{k}")
        else:
            data.extend([f"--{k}", json.dumps(v).replace('"', "")])
    return data



class TorchrunExecutor:

    """
    Instances of the TorchrunExecutor class are used to run the torchrun command.
    There is one per Metaflow @step annotated with @torchrun.

    TorchrunExecutor takes in information about this run based on the Metaflow config, 
    so users declare the infrastructure Metaflow dynamically spins up for them in one place, 
    and then the TorchrunExecutor uses that information to configure the distributed parts of the torchrun command accordingly.

    The Torchrun decorator, which users specify in a Metaflow num_parallel task with @torchrun, attaches an instance of this class to the current object.
    Using current.torch.run() will then run the torchrun command with the appropriate arguments.

    This class will handle opening the subprocess, and ensuring other typical Metaflow functionality works as expected.
    """

    def __init__(
        self, pathspec, main_addr, main_port, num_nodes, node_index, nproc_per_node=1, flow_datastore=None,
    ) -> None:
        self.torchrun_args = {
            "nnodes": num_nodes,
            "master_addr": main_addr,
            "master_port": main_port,
            "node_rank": node_index,
            "nproc_per_node": nproc_per_node,
        }
        self._flow_datastore = flow_datastore
        # TODO: heartbeat

    def _exec_cmd(
        self,
        torchrun_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
        nproc_per_node: int = 1
    ):
    
        self._ensure_torch_installed()
 
        # Container to build up the command to be run in a subprocess.
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc_per_node}"]

        # Construct the torchrun distributed arguments.

        # Group 1: args user supplied in torch.current.run
        if type(torchrun_args) == dict:
            self.torchrun_args.update(torchrun_args)
            torchrun_args = _dict_to_args(self.torchrun_args)
        cmd.extend(torchrun_args)

        # Group 2: defaults user did not specify
        for key, value in self.torchrun_args.items():
            if f'--{key}' not in cmd:
                cmd.extend([f'--{key}', str(value)])
        cmd.extend(["--rdzv_endpoint", "%s:%s" % (self.torchrun_args["master_addr"], self.torchrun_args["master_port"])])

        # Construct rest of command starting with the entrypoint.
        if entrypoint is not None:
            cmd.append(entrypoint)
        else:
            raise MetaflowException(
                "current.torchrun.run(..., entrypoint=<SCRIPT>, ...) arg must be specified."
            )
        if entrypoint_args is not None and isinstance(entrypoint_args, dict):
            cmd.extend(_dict_to_args(entrypoint_args))
        elif entrypoint_args is not None and isinstance(entrypoint_args, list):
            cmd.extend(entrypoint_args)

        # Launch the Torchrun process and stream logs.
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            while process.poll() is None:
                stdout = process.stdout.read1()
                try:
                    text = stdout.decode("utf-8")
                except UnicodeDecodeError:
                    text = ""
                print(
                    text, end="", flush=True
                )
            if process.returncode != 0:
                return False, "Process exited with errors (see above for details)"
            return True, None

    def _ensure_torch_installed(self):
        try:
            import torch
        except ImportError:
            raise TorchNotInstalledException()

    def _resolve_storage_paths(
        self, push_results_dir_to_cloud, datastore, local_output_dir, cloud_output_dir
    ) -> str:
        if push_results_dir_to_cloud and (
            datastore._backend.TYPE != "s3" and datastore._backend.TYPE != "azure"
        ):
            raise MetaflowException(
                "current.torchrun.run must use S3 or Azure Blob Storage as a datastore if push_results_dir_to_cloud is True. You are using %s."
                % datastore._backend.TYPE
            )
        elif push_results_dir_to_cloud:
            # TODO: Annoying place for this check. Consider moving the S3 push args into the decorator itself, so can be checked at flow init instead.
            if local_output_dir is None:
                raise MetaflowException(
                    "current.torchrun.run must specify local_output_dir if push_results_dir_to_s3 is True"
                )
            elif cloud_output_dir is None:
                if local_output_dir.startswith("/"):
                    cloud_output_dir = local_output_dir[1:]
                else:
                    cloud_output_dir = local_output_dir
        return cloud_output_dir

    def run(
        self,
        torchrun_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
        nproc_per_node: int = 1,
        push_results_dir_to_cloud: bool = False,
        local_output_dir: str = None,
        cloud_output_dir: str = None,
    ) -> Union[str, None]:
        from metaflow import current

        node_index = current.parallel.node_index  # assumes parallel
        datastore = TorchrunDatastore(
            flow_datastore=self._flow_datastore, pathspec=current.pathspec
        )

        # Resolve storage paths
        cloud_output_dir = self._resolve_storage_paths(
            push_results_dir_to_cloud, datastore, local_output_dir, cloud_output_dir
        )

        def _monitor_node_health(status_notifier, frequency=10):
            '''
            A naive monitor that checks the status of the torchrun processes of other nodes.
            If any other node has failed, this node should fail as soon as possible to free up resources.

            For complex fault tolerance patterns, there needs to be a more sophisticated health monitor that knows about Metaflow @retry.
            '''
            while status_notifier.read(node_index).status == "running":
                other_failed_nodes = []
                for i in range(current.parallel.num_nodes):
                    if i != node_index and status_notifier.read(i).status == "failed": 
                        other_failed_nodes.append(i) 
                if len(other_failed_nodes) > 0: 
                    status_notifier.failed(node_index)
                    raise MetaflowException(
                        f"current.torchrun.run on node {node_index} is being forcibly failed because other nodes have failed." 
                    )
                time.sleep(frequency)
  
        _status_notifier = TaskStatusNotifier(datastore)
        _status_notifier.running(node_index)
        _node_health_monitor = Thread(target=_monitor_node_health, args=(_status_notifier,))
        _node_health_monitor.start()
        success_status, stderr = self._exec_cmd(
            torchrun_args=torchrun_args,
            entrypoint=entrypoint,
            entrypoint_args=entrypoint_args,
            nproc_per_node=nproc_per_node,
        )
        if success_status:
            _status_notifier.finished(node_index)
        else:
            _status_notifier.failed(node_index)
            msg = f"The `torchrun` command running on node {node_index} has crashed. \n\n[stderr]: {str(stderr)}"
            raise TorchrunException(msg)
        _node_health_monitor.join()
        
        # Push results to S3
        # TODO: deprecate this completely in favor of @checkpoint and @model best practices 
        # if push_results_dir_to_cloud:
        #     if not os.path.exists(local_output_dir):
        #         print(
        #             f"Torchrun process completed, and local_output_dir `{local_output_dir}` does not exist, skipping push to datastore.",
        #             file=sys.stderr,
        #         )
        #         return None
        #     paths = datastore.put_files(
        #         _get_path(local_output_dir, cloud_output_dir, node_index)
        #     )
        #     print(
        #         f"Pushed {len(paths)} files to {datastore._backend.TYPE} at {cloud_output_dir}",
        #         file=sys.stderr,
        #     )
        #     return datastore.get_datastore_file_location(cloud_output_dir)


class TorchrunSingleNodeMultiGPU:

    """
    A slimmed down version of TorchrunExecutor, to be exposed to Metaflow user code for single node, multi-gpu use cases.
    """

    def __init__(self):
        pass

    def _exec_cmd(
        self,
        torchrun_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
        nproc_per_node: int = 1
    ):
        self._ensure_torch_installed()
 
        # Container to build up the command to be run in a subprocess.
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc_per_node}"]

        # Construct the torchrun distributed arguments.

        # Group 1: args user supplied in torch.current.run
        if type(torchrun_args) == dict:
            torchrun_args = _dict_to_args(torchrun_args)
        cmd.extend(torchrun_args)

        # Construct rest of command starting with the entrypoint.
        if entrypoint is not None:
            cmd.append(entrypoint)
        else:
            raise MetaflowException(
                "TorchrunSingleNodeMultiGPU().run(..., entrypoint=<SCRIPT>, ...) arg must be specified."
            )
        if entrypoint_args is not None and isinstance(entrypoint_args, dict):
            cmd.extend(_dict_to_args(entrypoint_args))
        elif entrypoint_args is not None and isinstance(entrypoint_args, list):
            cmd.extend(entrypoint_args)

        # Launch the Torchrun process and stream logs.
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            while process.poll() is None:
                stdout = process.stdout.read1()
                try:
                    text = stdout.decode("utf-8")
                except UnicodeDecodeError:
                    text = ""
                print(
                    text, end="", flush=True
                )
            if process.returncode != 0:
                return False, "Process exited with errors (see above for details)"
            return True, None

    def _ensure_torch_installed(self):
        try:
            import torch
        except ImportError:
            raise TorchNotInstalledException()

    def run(
        self,
        torchrun_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
        nproc_per_node: int = 1
    ) -> Union[str, None]:

        success_status, stderr = self._exec_cmd(
            torchrun_args=torchrun_args,
            entrypoint=entrypoint,
            entrypoint_args=entrypoint_args,
            nproc_per_node=nproc_per_node
        )
        if not success_status:
            msg = f"The `torchrun` command has crashed. \n\n[stderr]: {str(stderr)}"
            raise TorchrunException(msg)