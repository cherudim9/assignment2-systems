import pandas as pd
import triton
import torch
import torch._functorch.config
from cs336_systems.flash_attention_triton import FlashAttentionTritonFunc
from tests.test_attention import _attention_and_lse


torch._functorch.config.donated_buffer = False
DEVICE = 'cuda'
REP = 300
WARMUP = 100
N_HEADS = 4
model_list = [
    ('standard', lambda *args: _attention_and_lse(*args)[0]),
    ('my triton', FlashAttentionTritonFunc.apply)
]


def test_timing_flash_forward_backward(
    model,
    n_heads,
    d_head,
    sequence_length,
    dtype,
):
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device=DEVICE, dtype=dtype, requires_grad=True)
    
    model = torch.compile(model)

    def test_forward_only():
        model(q, k, v, True)
    forward_results = triton.testing.do_bench(test_forward_only, rep=REP, warmup=WARMUP)

    o = model(q, k, v, True)
    loss = o.sum()
    def test_backward_only():
        loss.backward(retain_graph=True)
    backward_results = triton.testing.do_bench(test_backward_only, rep=REP, warmup=WARMUP)

    def test_all():
        o = model(q, k, v, True)
        loss = o.sum()
        loss.backward()
    all_results = triton.testing.do_bench(test_all, rep=REP, warmup=WARMUP)
    
    return forward_results, backward_results, all_results
    

def test():
    results = []
    for d_head_i in range(4, 8):
        for sequence_length_i in range(7, 17):
            for dtype in [torch.float32, torch.bfloat16]:
                for model_name, model in model_list:
                    d_head = 2 ** d_head_i
                    sequence_length = 2 ** sequence_length_i
                    print(f'testing: model={model_name}, d_head={d_head}, sequence_length={sequence_length}, dtype={dtype}...')
                    res = test_timing_flash_forward_backward(
                        model=model,
                        n_heads=N_HEADS,
                        sequence_length=sequence_length,
                        d_head=d_head,
                        dtype=dtype,
                    )
                    results.append({
                        'model_name': model_name,
                        'sequence_length': sequence_length,
                        'd_head': d_head,
                        'dtype': str(dtype),
                        'forward_time': res[0],
                        'backward_time': res[1],
                        'all_time': res[2],
                    })
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("benchmark_results.csv", index=False)


if __name__ == "__main__":
    test()