from metaflow import FlowSpec, step, resources

class BertTrainingSingleNodeMultiGPU(FlowSpec):

    @resources(gpu=8)
    @step
    def start(self):
        from metaflow.plugins.torchrun_libs.executor import TorchrunSingleNodeMultiGPU

        executor = TorchrunSingleNodeMultiGPU()
        executor.run(entrypoint="train_script.py", nproc_per_node=8)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    BertTrainingSingleNodeMultiGPU()