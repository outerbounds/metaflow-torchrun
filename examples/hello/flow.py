from metaflow import FlowSpec, step, kubernetes, environment, torchrun, current


class HelloTorchrun(FlowSpec):
    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=2)

    @kubernetes(
        image="registry.hub.docker.com/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        cpu=2,
    )
    @environment(vars={"OMP_NUM_THREADS": 1})
    @torchrun
    @step
    def torch_multinode(self):
        current.torch.run(entrypoint="hi-torchrun.py")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    HelloTorchrun()
