from metaflow import FlowSpec, step, kubernetes, torchrun, current


class TorchrunTensorPass(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=2)

    @kubernetes(
        image="registry.hub.docker.com/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        gpu=1,
    )
    @torchrun
    @step
    def torch_multinode(self):
        current.torch.run(entrypoint="script.py")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TorchrunTensorPass()
