import math
import torch
import triton
import triton.language as tl
from einops import rearrange, einsum


Bq = 32
Bk = 32


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


@triton.jit
def flash_bwd_kernel_pass1(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr, mask_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_dok, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_mq, stride_mk,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    nq, nk,
    scale, 
    D: tl.constexpr,
    Bq: tl.constexpr, Bk: tl.constexpr,
    is_causal: tl.constexpr,
):
    batch_index = tl.program_id(0)
    key_tile_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(nk, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * Bk, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )
    kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(nk, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * Bk, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )
    vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(nq, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(nq, D),
        strides=(stride_dok, stride_dod),
        offsets=(0, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(nq,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Bq,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(nq,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Bq,),
        order=(0,),
    )

    mask_block_ptr = tl.make_block_ptr(
        mask_ptr,
        shape=(nq, nk),
        strides=(stride_mq, stride_mk),
        offsets=(0, key_tile_index * Bk),
        block_shape=(Bq, Bk),
        order=(1, 0),
    )

    dK_sum = tl.zeros((Bk, D), dtype=tl.float32)
    dV_sum = tl.zeros((Bk, D), dtype=tl.float32)

    for _ in range(tl.cdiv(nq, Bq)):
        qi =  tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        s = tl.dot(qi, kj.trans()) * scale
        if is_causal:
            mask = tl.load(mask_block_ptr, boundary_check=(0,1), padding_option="zero")
            s = tl.where(mask, s, -1e6)

        li =  tl.load(L_block_ptr, boundary_check=(0,1), padding_option="zero")
        p = tl.exp(s - li.expand_dims(axis=1))
        dV_sum = tl.dot(p.trans(), doi, acc=dV_sum)

        doi = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero")
        dP = tl.dot(doi, vj.trans())

        Di = tl.load(D_block_ptr, boundary_check=(0,1), padding_option="zero")
        ds = p * (dP - Di.expand_dims(axis=1)) * scale
        dK_sum = tl.dot(ds.trans(), qi, acc=dK_sum)

        Q_block_ptr = Q_block_ptr.advance((Bq, 0))
        dO_block_ptr = dO_block_ptr.advance((Bq, 0))
        L_block_ptr = L_block_ptr.advance((Bq,))
        D_block_ptr = D_block_ptr.advance((Bq,))
        mask_block_ptr = mask_block_ptr.advance((Bq, 0))

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(nk, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * Bk, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )
    tl.store(dK_block_ptr, dK_sum.to(dK_block_ptr.dtype.element_ty), boundary_check=(0,1))

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(nk, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * Bk, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )
    tl.store(dV_block_ptr, dV_sum.to(dV_block_ptr.dtype.element_ty), boundary_check=(0,1))


class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_shape = Q.shape
        K_shape = K.shape
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
        L = torch.empty((bs, N_QUERIES,), dtype=Q.dtype, device=Q.device)
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
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.Bq = Bq
        ctx.Bk = Bk
        ctx.Q_shape = Q_shape
        ctx.K_shape = K_shape
        ctx.is_causal = is_causal

        O = O.view(Q_shape)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        # recover all parameters
        _, nq, d = Q.shape
        nk = K.shape[1]
        scale = d ** -0.5

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        mask = (torch.arange(nq)[:, None] >= torch.arange(nk)[None, :]).to(Q.device)

        S = einsum(Q, K, "bs nq d, bs nk d -> bs nq nk") * scale
        if ctx.is_causal:
            S = torch.where(mask[None, :], S, float("-inf"))
        P = torch.exp(S - L[:, :, None])
        dV = einsum(P, dO, "bs nq nk, bs nq d -> bs nk d")
        dP = einsum(dO, V, "bs nq d, bs nk d -> bs nq nk")

        D = torch.sum(O * dO, dim=-1)
        dS = P * (dP - D[:, :, None])

        dQ = einsum(dS, K, "bs nq nk, bs nk d -> bs nq d") * scale
        dK = einsum(dS, Q, "bs nq nk, bs nq d -> bs nk d") * scale

        dQ = dQ.view(ctx.Q_shape)
        dK = dK.view(ctx.K_shape)
        dV = dV.view(ctx.K_shape)
        return dQ, dK, dV, None

    @staticmethod
    def backward_triton(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        # recover all parameters
        Bq, Bk = ctx.Bq, ctx.Bk
        bs, nq, d = Q.shape
        nk = K.shape[1]
        Tq = math.ceil(nq/Bq)
        Tk = math.ceil(nk/Bk)
        scale = d ** -0.5
        is_causal = ctx.is_causal

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        D = torch.sum(O * dO, dim=-1)
        mask = (torch.arange(nq)[:, None] >= torch.arange(nk)[None, :]).to(Q.device)
        flash_bwd_kernel_pass1[(bs, Tk)](
            Q, K, V,
            dO, L, D, mask,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            mask.stride(0), mask.stride(1), mask.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            nq=nq, nk=nk,
            scale=scale,
            D=d,
            Bq=Bq, Bk=Bk,
            is_causal=is_causal,
        )

        dQ = dQ.view(ctx.Q_shape)
        dK = dK.view(ctx.K_shape)
        dV = dV.view(ctx.K_shape)
        return dQ, dK, dV, None
    
    @staticmethod
    def backward_triton_in_pytorch(ctx, dO):
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
        mask = (torch.arange(nq)[:, None] >= torch.arange(nk)[None, :]).to(Q.device)
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
                    if ctx.is_causal:
                        s = torch.where(
                            mask[i * Bq : min((i + 1) * Bq, nq), j * Bk : min((j + 1) * Bk, nk)],
                            s, -1e6)

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