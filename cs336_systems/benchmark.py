import triton
import torch
from cs336_systems.flash_attention_triton import FlashAttentionTritonFunc


DEVICE = 'mps'


def test_timing_flash_forward_backward(
    model,
    n_heads,
    d_head,
    sequence_length,
    dtype,
):
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device=DEVICE, dtype=dtype, requires_grad=True)
    
    if DEVICE == 'mps':
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)

    def test_forward_only():
        o = model(q, k, v, True)
    results = triton.testing.do_bench(test_forward_only, rep=1000, warmup=100)
    print(results)
    

def test():
    test_timing_flash_forward_backward(
        model=FlashAttentionTritonFunc.apply,
        n_heads=16,
        sequence_length=128,
        d_head=16,
        dtype=torch.float32,
    )


if __name__ == "__main__":
    test()