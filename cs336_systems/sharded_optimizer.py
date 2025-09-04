import torch
from typing import Type, Any
import torch.distributed as dist

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs: Any
    ):
        defaults = {}
        self.all_params = list(params)
        self.shard = []
        for i, p in enumerate(self.all_params):
            p.group = i % dist.get_world_size()
            if p.group == dist.get_rank():
                self.shard.append(p)

        self.optim = optimizer_cls(self.shard, **kwargs)
        super().__init__(self.shard, defaults)

    def step(self, closure=None, **kwargs):
        loss = self.optim.step(closure, **kwargs)
        for p in self.all_params:
            dist.broadcast(p.data, src=p.group, async_op=False)
        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)