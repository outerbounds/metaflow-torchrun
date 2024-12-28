from metaflow import FlowSpec, Parameter, step, kubernetes, environment, torchrun, current


class HelloTorchrun(FlowSpec):
    num_nodes = Parameter(
        name="num_nodes",
        default=2,
        type=int,
        required=True,
        help="num_nodes"
    )

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=self.num_nodes)

    @kubernetes(
        image="registry.hub.docker.com/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        cpu=2,
    )
    # to remove the following warning, but not strictly required:
    ## Setting OMP_NUM_THREADS environment variable for each process to be 1 in default,
    ## to avoid your system being overloaded, please further tune the variable for optimal
    ## performance in your application as needed.
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
