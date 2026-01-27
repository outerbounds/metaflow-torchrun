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


def _list_to_args(l: List[str]) -> List[str]:
    data = []

    for idx, item in enumerate(l):
        if idx % 2 == 0:
            # arg
            data.append(f"--{item}")
        else:
            data.append(json.dumps(item).replace('"', ""))
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
        
        # These arguments are pass upon attaching to the metaflow.current.torch object.
        # Values may be overridden by a user passing values to metaflow.current.torch.run's torchrun_args.
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
        torchrun_args: Union[List[str], Dict[str, str]] = {},
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = {},
        nproc_per_node: int = None
    ):
        # nproc_per_node is a special parameter, often featured in torchrun commands across the internet.
        # It is the only arg that can be specified with its own parameter value, or within torchrun_args.
        # This block processes accordingly to find a single value. The order of precedence:
            # 1. user override --> nproc_per_node argument of current.torch.run.
            # 2. user override --> 'nproc_per_node' key of torchrun_args arg of current.torch.run.
            # 3.       default --> 1. NOTE: If desired in future, this could be derived from @resources spec, see torchrun_decorator.py.
        if nproc_per_node is not None and 'nproc_per_node' in torchrun_args:
            # executor.run(entrypoint="script.py", nproc_per_node=4, torchrun_args={"nproc_per_node": 3}) XXX INVALID XXX
            error_msg = 'If nproc_per_node argument in current.torch.run is specified, do not set nproc_per_node in torchrun_args.'
            return False, error_msg
        elif nproc_per_node is None and 'nproc_per_node' not in torchrun_args:
            # executor.run(entrypoint="script.py") XXX DEFAULT = 1 XXX
            if isinstance(torchrun_args, dict):
                torchrun_args['nproc_per_node'] = str(1)
            elif isinstance(torchrun_args, list):
                try:
                    idx = torchrun_args.index('nproc_per_node')
                    if len(torchrun_args) < idx + 1:
                        error_msg = 'If torchrun_args is a list and nproc_per_node is present, it needs to have a value following it in the list.'
                        return False, error_msg
                    torchrun_args[idx+1] = str(1)
                except ValueError: 
                    torchrun_args.extend(['nproc_per_node', str(1)])
        elif nproc_per_node is None and 'nproc_per_node' in torchrun_args:
            # executor.run(entrypoint="script.py", torchrun_args={"nproc_per_node": 4}) XXX USER OPTION = 4 XXX
            if isinstance(torchrun_args, dict):
                pass
            # executor.run(entrypoint="script.py", torchrun_args=["nproc_per_node", 4]) XXX USER OPTION = 4 XXX
            elif isinstance(torchrun_args, list):
                idx = torchrun_args.index('nproc_per_node')
                if len(torchrun_args) < idx+1:
                    error_msg = 'If torchrun_args is a list and nproc_per_node is present, it needs to have a value following it in the list.'
                    return False, error_msg
                torchrun_args[idx+1] = str(torchrun_args[idx+1])
        else:
            # executor.run(entrypoint="script.py", nproc_per_node=6, torchrun_args={...}) XXX USER OPTION = 6 XXX
            if isinstance(torchrun_args, dict):
                torchrun_args['nproc_per_node'] = nproc_per_node
            elif isinstance(torchrun_args, list):
                idx = torchrun_args.get('nproc_per_node')
                if len(torchrun_args) < idx+1:
                    error_msg = 'If torchrun_args is a list and nproc_per_node is present, it needs to have a value following it in the list.'
                    return False, error_msg
                torchrun_args[idx+1] = str(torchrun_args[idx+1])
    
        self._ensure_torch_installed()
 
        # Container to build up the command to be run in a subprocess.
        cmd = [sys.executable, "-m", "torch.distributed.run"]

        # Construct the torchrun distributed arguments.

        # Group 1: args user supplied in torch.current.run
        if type(torchrun_args) == dict:
            self.torchrun_args.update(torchrun_args)
            torchrun_args = _dict_to_args(self.torchrun_args)
        elif isinstance(torchrun_args, list):
            torchrun_args = _list_to_args(torchrun_args)
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
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as process:
                while process.poll() is None:
                    stdout = process.stdout.read1()
                    stderr = process.stderr.read1()
                    try:
                        stdout_text = stdout.decode("utf-8")
                    except UnicodeDecodeError:
                        stdout_text = ""
                    try:
                        stderr_text = stderr.decode("utf-8")
                    except UnicodeDecodeError:
                        stderr_text = ""
                    print(stdout_text, end="", flush=True)
                    print(stderr_text, end="", flush=True, file=sys.stderr)
                if process.returncode != 0:
                    raise TorchrunException(f"Process exited with return code {process.returncode}")
                return True, None
        except (FileNotFoundError, PermissionError, OSError) as e:
            return False, f"Failed to start subprocess: {str(e)}"


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
        torchrun_args: Union[List[str], Dict[str, str]] = {},
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = {},
        nproc_per_node: int = None,
        push_results_dir_to_cloud: bool = False,
        local_output_dir: str = None,
        cloud_output_dir: str = None,
    ) -> Union[str, None]:
        from metaflow import current

        node_index = current.parallel.node_index  # assumes parallel
        datastore = TorchrunDatastore(
            flow_datastore=self._flow_datastore, pathspec=current.pathspec
        )

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
        try:
            success_status, stderr = self._exec_cmd(
                torchrun_args=torchrun_args,
                entrypoint=entrypoint,
                entrypoint_args=entrypoint_args,
                nproc_per_node=nproc_per_node,
            )
            if success_status:
                _status_notifier.finished(node_index)
            else:
                # This handles cases where subprocess failed to start (FileNotFoundError, etc.)
                _status_notifier.failed(node_index)
                msg = f"The `torchrun` command running on node {node_index} has crashed. \n\n[stderr]: {str(stderr)}"
                raise TorchrunException(msg)
        except TorchrunException:
            # Process failed during execution (non-zero return code)
            _status_notifier.failed(node_index)
            raise
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
        torchrun_args: Union[List[str], Dict[str, str]] = {},
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = {},
        nproc_per_node: int = None
    ):
        # nproc_per_node is a special parameter, often featured in torchrun commands across the internet.
        # It is the only arg that can be specified with its own parameter value, or within torchrun_args.
        # This block processes accordingly to find a single value. The order of precedence:
            # 1. user override --> nproc_per_node argument of current.torch.run.
            # 2. user override --> 'nproc_per_node' key of torchrun_args arg of current.torch.run.
            # 3.       default --> 1. NOTE: If desired in future, this could be derived from @resources spec, see torchrun_decorator.py.
        if nproc_per_node is not None and 'nproc_per_node' in torchrun_args:
            # executor.run(entrypoint="script.py", nproc_per_node=4, torchrun_args={"nproc_per_node": 3}) XXX INVALID XXX
            error_msg = 'If nproc_per_node argument of TorchrunSingleNodeMultiGPU().run is specified, do not set nproc_per_node in torchrun_args.'
            print(f'[current.torch.run ERROR] {error_msg}')
            return False, error_msg
        elif nproc_per_node is None and 'nproc_per_node' not in torchrun_args:
            # executor.run(entrypoint="script.py") XXX DEFAULT = 1 XXX
            if isinstance(torchrun_args, dict):
                torchrun_args['nproc_per_node'] = str(1)
            elif isinstance(torchrun_args, list):
                try:
                    idx = torchrun_args.index('nproc_per_node')
                    if len(torchrun_args) < idx + 1:
                        error_msg = 'If torchrun_args is a list and nproc_per_node is present, it needs to have a value following it in the list.'
                        return False, error_msg
                    torchrun_args[idx+1] = str(1)
                except ValueError:
                    torchrun_args.extend(['nproc_per_node', str(1)])
        elif nproc_per_node is None and 'nproc_per_node' in torchrun_args:
            # executor.run(entrypoint="script.py", torchrun_args={"nproc_per_node": 4}) XXX USER OPTION = 4 XXX
            if isinstance(torchrun_args, dict):
                pass
            # executor.run(entrypoint="script.py", torchrun_args=["nproc_per_node", 4]) XXX USER OPTION = 4 XXX
            elif isinstance(torchrun_args, list):
                idx = torchrun_args.index('nproc_per_node')
                if len(torchrun_args) < idx+1:
                    error_msg = 'If torchrun_args is a list and nproc_per_node is present, it needs to have a value following it in the list.'
                    return False, error_msg
                nproc_per_node = str(torchrun_args[idx+1])
                torchrun_args[idx+1] = nproc_per_node
        else:
            # executor.run(entrypoint="script.py", nproc_per_node=6, torchrun_args={...}) XXX USER OPTION = 6 XXX
            if isinstance(torchrun_args, dict):
                torchrun_args['nproc_per_node'] = nproc_per_node
            elif isinstance(torchrun_args, list):
                try:
                    idx = torchrun_args.index('nproc_per_node')
                
                    if len(torchrun_args) < idx+1:
                        error_msg = 'If torchrun_args is a list and nproc_per_node is present, it needs to have a value following it in the list.'
                        return False, error_msg
                    torchrun_args[idx+1] = str(torchrun_args[idx+1])
                except ValueError:
                    # torchrun_args does not contain nproc_per_node
                    torchrun_args.extend(['nproc_per_node', str(nproc_per_node)])

        self._ensure_torch_installed()
 
        # Container to build up the command to be run in a subprocess.
        cmd = [sys.executable, "-m", "torch.distributed.run"]

        # Construct the torchrun distributed arguments.

        # Group 1: args user supplied in torch.current.run
        if isinstance(torchrun_args, dict):
            torchrun_args = _dict_to_args(torchrun_args)
        elif isinstance(torchrun_args, list):
            torchrun_args = _list_to_args(torchrun_args)
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
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as process:
                while process.poll() is None:
                    stdout = process.stdout.read1()
                    stderr = process.stderr.read1()
                    try:
                        stdout_text = stdout.decode("utf-8")
                    except UnicodeDecodeError:
                        stdout_text = ""
                    try:
                        stderr_text = stderr.decode("utf-8")
                    except UnicodeDecodeError:
                        stderr_text = ""
                    print(stdout_text, end="", flush=True)
                    print(stderr_text, end="", flush=True, file=sys.stderr)
                if process.returncode != 0:
                    raise TorchrunException(f"Process exited with return code {process.returncode}")
                return True, None
        except (FileNotFoundError, PermissionError, OSError) as e:
            return False, f"Failed to start subprocess: {str(e)}"

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
        nproc_per_node: int = None
    ) -> Union[str, None]:

        success_status, stderr = self._exec_cmd(
            torchrun_args=torchrun_args,
            entrypoint=entrypoint,
            entrypoint_args=entrypoint_args,
            nproc_per_node=nproc_per_node
        )
        if not success_status:
            # This handles cases where subprocess failed to start (FileNotFoundError, etc.)
            msg = f"The `torchrun` command has crashed. \n\n[stderr]: {str(stderr)}"
            raise TorchrunException(msg)
        # If process failed during execution (non-zero return code), TorchrunException is already raised