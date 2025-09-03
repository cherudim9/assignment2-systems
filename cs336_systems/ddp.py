import torch
import torch.distributed as dist


def param_update_fn(param: torch.Tensor):
    param.handle = dist.all_reduce(param.grad, dist.ReduceOp.SUM, async_op=True)


class DdpIndy(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            dist.broadcast(p.data, 0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(param_update_fn)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for p in self.module.parameters():
            if p.requires_grad and p.handle:
                p.handle.wait()
                p.grad = p.grad / dist.get_world_size()


def ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    return DdpIndy(module)
