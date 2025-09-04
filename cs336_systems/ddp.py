import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


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


def ddp_param_update_fn(param: torch.Tensor):
    param.ready = 1
    cnt = sum([p.ready for p in param.bucket])
    if cnt == len(param.bucket):
        param.fg = _flatten_dense_tensors([p.grad for p in param.bucket])
        param.handle = dist.all_reduce(param.fg, dist.ReduceOp.SUM, async_op=True)


class DdpBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module

        params_list = list(self.parameters())
        self.param_buckets = []
        current_bucket = []
        current_bucket_size = 0
        print(f'bucket_size_mb={bucket_size_mb}')
        for idx, p in enumerate(reversed(params_list)):
            dist.broadcast(p.data, 0)
            if p.requires_grad:
                p_size = p.data.numel() * p.element_size() / 1e6
                print(f'p_{idx}, size={p_size}')
                size_to_be = current_bucket_size + p_size
                if size_to_be > bucket_size_mb:
                    self.param_buckets.append(current_bucket)
                    print(f'new bucket of size = {current_bucket_size: .6f}')
                    current_bucket_size = 0
                    current_bucket = []
                    size_to_be = p_size
                current_bucket_size = size_to_be
                current_bucket.append(p)

        if len(current_bucket) > 0:
            self.param_buckets.append(current_bucket)
            print(f'new bucket of size = {current_bucket_size: .6f}')

        for pb in self.param_buckets:
            for p in pb:
                p.bucket = pb
                p.register_post_accumulate_grad_hook(ddp_param_update_fn)

    def reset_gradient(self):
        for pb in self.param_buckets:
            for p in pb:
                p.ready = 0
                p.handle = None

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for p in self.module.parameters():
            if p.requires_grad and p.handle:
                p.handle.wait()
                orig_g = _unflatten_dense_tensors(p.fg, [pi.grad for pi in p.bucket])
                for og, pi in zip(orig_g, p.bucket):
                    pi.grad = og / dist.get_world_size()


def ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    return DdpBucketed(module, bucket_size_mb)
