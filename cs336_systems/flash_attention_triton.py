import math
import torch
import triton
import triton.language as tl
from einops import rearrange


Bq = 16
Bk = 16


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    mask_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_mq, stride_mk,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):

    batch_index = tl.program_id(0)
    query_tile_index = tl.program_id(1)
    
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

    mask_block_ptr = tl.make_block_ptr(
        mask_ptr,
        shape=(N_QUERIES, N_KEYS),
        strides=(stride_mq, stride_mk),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, K_TILE_SIZE),
        order=(1, 0),
    )

    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    
    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")

        s = tl.dot(qi, kj.trans()) * scale
        if is_causal:
            mask = tl.load(mask_block_ptr, boundary_check=(0,1), padding_option="zero")
            s = tl.where(mask, s, -1e6)
        prev_m = m
        m = tl.maximum(prev_m, tl.max(s, axis=1))
        p = tl.exp(s - m.expand_dims(axis=1))
        l = l * tl.exp(prev_m - m) + tl.sum(p, axis=1)
        o = tl.exp(prev_m - m).expand_dims(axis=1) * o
        o = tl.dot(p.to(dtype=V_block_ptr.dtype.element_ty), vj, acc=o)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        mask_block_ptr = mask_block_ptr.advance((0, K_TILE_SIZE))

    o = (1.0 / l).expand_dims(axis=1) * o
    l = tl.log(l) + m

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, o.to(O_block_ptr.dtype.element_ty), boundary_check=(0,1))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    tl.store(L_block_ptr, l.to(L_block_ptr.dtype.element_ty), boundary_check=(0,))


class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_shape = Q.shape
        ctx.q_shape = Q_shape
        ctx.is_causal = is_causal
        Q = rearrange(Q, "... seq_len d -> (...) seq_len d")
        K = rearrange(K, "... seq_len d -> (...) seq_len d")
        V = rearrange(V, "... seq_len d -> (...) seq_len d")
        
        bs, N_QUERIES, D = Q.shape
        scale = D ** -0.5
        Tq = math.ceil(N_QUERIES / Bq)
        bs, N_KEYS, D = K.shape

        O = torch.empty_like(Q)
        L = torch.empty((bs, N_QUERIES,), device=Q.device)
        mask = torch.arange(N_QUERIES, device=Q.device)[:, None] >= torch.arange(N_KEYS, device=Q.device)[None, :]

        flash_fwd_kernel[(bs, Tq)](
            Q, K, V,
            O, L,
            mask,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            mask.stride(0), mask.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=D,  
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )

        L = L.view(Q_shape[:-1])
        ctx.save_for_backward(L)

        O = O.view(Q_shape)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        # recover all parameters
        Bq, Bk = ctx.Bq, ctx.Bk
        bs, nq, d = Q.shape
        nk = K.shape[1]
        Tq = math.ceil(nq/Bq)
        Tk = math.ceil(nk/Bk)
        isrd = d ** -0.5

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        for batch_idx in range(bs):
            D = torch.sum(O[batch_idx] * dO[batch_idx], dim=1)
            for j in range(Tk):
                kj = K[batch_idx, j * Bk : min((j + 1) * Bk, nk), :]
                vj = V[batch_idx, j * Bk : min((j + 1) * Bk, nk), :]
                dK_sum = torch.zeros_like(kj)
                dV_sum = torch.zeros_like(vj)
                for i in range(Tq):
                    qi = Q[batch_idx, i * Bq : min((i + 1) * Bq, nq), :]
                    doi = dO[batch_idx, i * Bq : min((i + 1) * Bq, nq), :]
                    li = L[batch_idx, i * Bq : min((i + 1) * Bq, nq)]
                    Di = D[i * Bq : min((i + 1) * Bq, nq)]

                    s = qi @ kj.T * isrd
                    p = torch.exp(s - li[:, None])
                    dV_sum += p.T @ doi
                    dP = doi @ vj.T
                    ds = p * (dP - Di[:, None]) * isrd
                    dQ[batch_idx, i * Bq : min((i + 1) * Bq, nq), :] += ds @ kj
                    dK_sum += ds.T @ qi

                dK[batch_idx, j * Bk : min((j + 1) * Bk, nk), :] = dK_sum
                dV[batch_idx, j * Bk : min((j + 1) * Bk, nk), :] = dV_sum

        dQ = dQ.view(ctx.Q_shape)
        dK = dK.view(ctx.K_shape)
        dV = dV.view(ctx.K_shape)
        return dQ, dK, dV, None


