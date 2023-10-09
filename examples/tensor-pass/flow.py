from metaflow import FlowSpec, step, torchrun_parallel, current, batch, kubernetes, environment

N_NODES = 2
N_GPU = 1

class CoreweaveTorchrunTensorPass(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @environment(vars = {"NCCL_SOCKET_IFNAME": "eth0"}) 
    @kubernetes(image="eddieob/hello-torchrun:12", gpu=N_GPU)
    @torchrun_parallel
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
    CoreweaveTorchrunTensorPass()