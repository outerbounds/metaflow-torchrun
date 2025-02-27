from metaflow import FlowSpec, step

class SingleNodeMultiGPUTorchrun(FlowSpec):

    @step
    def start(self):
        from metaflow import TorchrunSingleNodeMultiGPU

        executor = TorchrunSingleNodeMultiGPU()
        executor.run(entrypoint="script.py", nproc_per_node=2)
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    SingleNodeMultiGPUTorchrun()