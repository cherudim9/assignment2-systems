import math
import torch
import triton
import triton.language as tl
from einops import rearrange

Bq = 16
Bk = 16

class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_shape = Q.shape
        Q = rearrange(Q, "... seq_len d -> (...) seq_len d")
        K = rearrange(K, "... seq_len d -> (...) seq_len d")
        V = rearrange(V, "... seq_len d -> (...) seq_len d")
        
        bs, nq, d = Q.shape
        isrd = d ** -0.5
        Tq = math.ceil(nq/Bq)
        bs, nk, _ = K.shape
        Tk = math.ceil(nk/Bk)

        O = torch.empty_like(Q)
        L = torch.empty((bs, nq,))

        for batch_idx in range(bs):
            for i in range(Tq):
                qi = Q[batch_idx, i * Bq : min((i + 1) * Bq, nq), :]
                oi = torch.zeros_like(qi)
                li = torch.zeros((Bq,), device=Q.device)
                mi = torch.full((Bq,), float("-inf"), device=Q.device)
                for j in range(Tk):
                    kj = K[batch_idx, j * Bk : min((j + 1) * Bk, nk), :]
                    vj = V[batch_idx, j * Bk : min((j + 1) * Bk, nk), :]
                    s = qi @ kj.T * isrd

                    mi_jm1 = mi.clone()
                    mi = torch.maximum(mi_jm1, torch.max(s, dim=1).values)
                    pi = torch.exp(s - mi.unsqueeze(1))
                    li = li * torch.exp(mi_jm1 - mi) + torch.sum(pi, dim=1)
                    oi = torch.diag(torch.exp(mi_jm1 - mi)) @ oi + pi @ vj

                O[batch_idx, i * Bq : min((i + 1) * Bq, nq), :] = torch.diag(1.0 / li) @ oi
                L[batch_idx, i * Bq : min((i + 1) * Bq, nq)] = torch.log(li) + mi

        L = L.view(Q_shape[:-1])
        ctx.save_for_backward(L)

        O = O.view(Q_shape)
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index\
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")

        s = tl.dot(qi, kj.trans()) * scale
        prev_m = m
        m = tl.maximum(prev_m, tl.max(s, axis=1))
        p = tl.exp(s - m.expand_dims(axis=1))
        l = l * tl.exp(prev_m - m) + tl.sum(p, axis=1)
        o = (prev_m - m).expand_dims(axis=1) * o
        tl.dot(p.cast(dtype=V_block_ptr.dtype.element_ty), vj, acc=o)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    tl.store(O_block_ptr, o.cast(dtype=O_block_ptr.dtype.element_ty), boundary_check=(0,1))
    tl.store(L_block_ptr, l.cast(dtype=L_block_ptr.dtype.element_ty), boundary_check=(0,))


class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_shape = Q.shape
        Q = rearrange(Q, "... seq_len d -> (...) seq_len d")
        K = rearrange(K, "... seq_len d -> (...) seq_len d")
        V = rearrange(V, "... seq_len d -> (...) seq_len d")
        
        bs, N_QUERIES, D = Q.shape
        scale = D ** -0.5
        Tq = math.ceil(N_QUERIES / Bq)
        bs, N_KEYS, D = K.shape

        O = torch.empty_like(Q)
        L = torch.empty((bs, N_QUERIES,), device=Q.device)

        flash_fwd_kernel[(bs, Tq)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=D,  
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
        )

        L = L.view(Q_shape[:-1])
        ctx.save_for_backward(L)

        O = O.view(Q_shape)
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError



