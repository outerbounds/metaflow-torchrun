from metaflow.exception import MetaflowException


class TorchrunException(MetaflowException):
    headline = ""

    def __init__(self, cmd):
        msg = "The torchrun command \n\n{}\n\nfailed to complete.".format(cmd)
        super(TorchrunException, self).__init__(msg)

class TorchNotInstalledException(TorchrunException):
    headline = "PyTorch not installed"

    def __init__(self):
        msg = "PyTorch is not installed. Please install PyTorch before using the @torchrun decorator."
        super(TorchNotInstalledException, self).__init__(msg)


class DatastoreKeyNotFoundError(MetaflowException):
    headline = "DeepSpeed Datastore Not Found"

    def __init__(self, datastore_path_name):
        msg = "The DeepSpeed datastore path {} was not found.".format(
            datastore_path_name
        )
        super(DatastoreKeyNotFoundError, self).__init__(msg)


class BarrierTimeoutException(MetaflowException):
    headline = "Barrier Timeout"

    def __init__(self, lock_name, description):
        msg = f"Task has timed out after waiting for some keys to be written to the datastore.\n[Barrier Name]:{lock_name}\n[Barrier Info]: {description}"
        super(BarrierTimeoutException, self).__init__(msg)


class AllNodesStartupTimeoutException(MetaflowException):
    headline = "All workers did not join cluster error"

    def __init__(self):
        msg = "Exiting job due to time out waiting for all workers to join cluster. You can set the timeout in @torchrun(all_nodes_started_timeout=X)"
        super(AllNodesStartupTimeoutException, self).__init__(msg)

