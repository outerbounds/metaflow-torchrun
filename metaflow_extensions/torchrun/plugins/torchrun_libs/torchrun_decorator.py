from functools import partial
import subprocess
import socket
import time
import json
import sys
import os

from metaflow.plugins.parallel_decorator import (
    ParallelDecorator,
    _local_multinode_control_task_step_func,
    UBF_CONTROL,
)
from metaflow import current

from .exceptions import (
    TorchrunException,
    TorchNotInstalledException,
    AllNodesStartupTimeoutException,
)
from .executor import TorchrunExecutor
from .torch_distributed_setup import setup_torch_distributed_env


class TorchrunDecoratorParallel(ParallelDecorator):
    name = "torchrun"
    defaults = {
        "master_port": "3339",
        # TODO : Safe rename `all_nodes_started_timeout` to something more intuitive.
        "all_nodes_started_timeout": 600,
        "nproc_per_node": 1,
    }
    IS_PARALLEL = True

    def _setup_current(self, main_addr, main_port, ubf_context, num_nodes, node_index):
        current._update_env(
            {
                "torch": TorchrunExecutor(
                    pathspec=current.pathspec,
                    main_addr=main_addr,
                    main_port=main_port,
                    num_nodes=num_nodes,
                    node_index=node_index,
                    nproc_per_node=self.nproc_per_node,
                    flow_datastore=self.flow_datastore,
                )
            }
        )

    def step_init(self, flow, graph, step, decos, environment, flow_datastore, logger):

        self.flow_datastore = flow_datastore

        for deco in decos: 
            if deco.name in ["resources", "kubernetes", "batch"]:
                if 'trainium' in deco.attributes and deco.attributes['trainium'] != 0:
                    self.nproc_per_node = deco.attributes['trainium'] * 2 # each trainium/inferentia device has 2 cores
                elif 'inferentia' in deco.attributes and deco.attributes['inferentia'] != 0:
                    self.nproc_per_node = deco.attributes['inferentia'] * 2
                elif deco.attributes['gpu']:
                    self.nproc_per_node = deco.attributes['gpu']
                else:
                    self.nproc_per_node = deco.attributes['cpu']

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        self._ubf_context = ubf_context

    def setup_distributed_env(self, run):
        setup_torch_distributed_env(self.attributes["all_nodes_started_timeout"], self.flow_datastore)
        self._setup_current(
            current.parallel.main_ip,
            self.attributes["master_port"],
            self._ubf_context,
            current.parallel.num_nodes,
            current.parallel.node_index,
        )