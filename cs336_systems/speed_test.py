import pandas as pd
import triton
import torch
import torch._functorch.config
from cs336_systems.flash_attention_triton import FlashAttentionTritonFunc
from tests.test_attention import _attention_and_lse


DEVICE = 'cuda'
REP = 10000
WARMUP = 1000
N_HEADS = 16
D_HEAD = 1024
SEQUENCE_LENGTH = 16384


def test_timing_flash_forward_backward():
    q, k, v = torch.randn(
        3, N_HEADS, SEQUENCE_LENGTH, D_HEAD, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    
    model = torch.compile(FlashAttentionTritonFunc.apply)

    def test_all():
        o = model(q, k, v, True)
        loss = o.sum()
        loss.backward()

    ret = triton.testing.do_bench(test_all, rep=REP, warmup=WARMUP)
    
    return ret


if __name__ == "__main__":
    print(test_timing_flash_forward_backward())