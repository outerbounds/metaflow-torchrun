from metaflow import (
    FlowSpec,
    step,
    torchrun,
    current,
    kubernetes
)

N_NODES = 2
N_GPU = 1

class TorchrunDDP(FlowSpec):
    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @kubernetes(image="docker.io/eddieob/hello-torchrun:12", gpu=N_GPU)
    @torchrun
    @step
    def torch_multinode(self):
        current.torch.run(
            entrypoint="multinode_trainer.py",
            entrypoint_args={"total-epochs": 2, "batch-size": 32, "save-every": 1},
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TorchrunDDP()
