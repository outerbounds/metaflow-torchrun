
import json

from metaflow import current

from .datastore import TorchrunDatastore, task_sync_barrier
from .constants import NODE_STARTED_VAR

class KeyPaths:
    Node = lambda x: f"{NODE_STARTED_VAR}/node_{x}"


def setup_torch_distributed_env(
    all_nodes_started_timeout,
    flow_datastore,
    polling_frequency=0.1,
):

    """  
    Wait for the all workers to write the keys in the datastore.
    """

    datastore = TorchrunDatastore(flow_datastore=flow_datastore, pathspec=current.pathspec)
    key_push_path = KeyPaths.Node(current.parallel.node_index)
    datastore.put(key_push_path, json.dumps({"started": True}))
    _all_keys = [KeyPaths.Node(i) for i in range(current.parallel.num_nodes)]
    _lock_args = {
        'description': 'Waiting for all nodes to start',
        'max_wait_time': all_nodes_started_timeout,
        'frequency': polling_frequency,
    }
    with task_sync_barrier("AllNodesStartedSync", datastore, _all_keys, **_lock_args):
        pass # All nodes have started