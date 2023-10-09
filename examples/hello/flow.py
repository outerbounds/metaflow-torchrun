from metaflow import FlowSpec, step, torchrun_parallel, current, batch, kubernetes

N_NODES = 2

class HelloTorchrun(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @kubernetes(image="pytorch/pytorch:latest")
    @torchrun_parallel
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