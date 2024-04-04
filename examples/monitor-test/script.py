import torch.distributed as dist
import torch
import os
import time

if __name__ == "__main__":
    # initialize the process group
    dist.init_process_group(backend="gloo")

    torch.distributed.barrier()

    # get the rank and size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # a simple way to print a message from each process
    print("hello from node %s of %s" % (rank + 1, world_size))

    for i in range(6):
        if i == 2 and rank == 1:
            # Expect: Node 1 to fail at i=2
            # Expect: Control node (and all others if N>2) to fail at i=3 when its health monitor realizes that Node 1 has failed
            raise ValueError("This is a test error.")
        time.sleep(10)

    # tear down the process group
    dist.destroy_process_group()
