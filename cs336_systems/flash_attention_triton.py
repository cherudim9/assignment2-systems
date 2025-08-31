import math
import os
import torch
import triton
import triton.language as tl
from einops import rearrange, einsum


QUERY_BLOCK_SIZES = [32, 64, 128]
KEY_BLOCK_SIZES = [32, 64, 128]
NUM_STAGES = [2, 3, 4]
NUM_WARPS = [2, 4, 8]
if "PYTEST_VERSION" in os.environ:
    QUERY_BLOCK_SIZES = [32]
    KEY_BLOCK_SIZES = [32]
    NUM_STAGES = [2]
    NUM_WARPS = [1]


def autotune_get_configs(block_names):
    return [
        triton.Config({block_names[0]: query_block_size, block_names[1]: key_block_size}, num_stages=s, num_warps=w)
        for query_block_size in QUERY_BLOCK_SIZES
        for key_block_size in KEY_BLOCK_SIZES
        for s in NUM_STAGES
        for w in NUM_WARPS
    ]


def prune_invalid_configs(configs, named_args, **kwargs):
    nq = kwargs["nq"]
    nk = kwargs["nk"]
    return [conf for conf in configs if conf.kwargs.get("Bq", 0) <= nq and conf.kwargs.get("Bk", 0) <= nk]


@triton.autotune(
    configs=autotune_get_configs(['Bq', 'Bk']),
    key=['nq', 'nk', 'D'],
    prune_configs_by={'early_config_prune': prune_invalid_configs}
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    nq: tl.constexpr,
    nk: tl.constexpr,
    D: tl.constexpr,
    Bq: tl.constexpr,
    Bk: tl.constexpr,
    is_causal: tl.constexpr,
):

    batch_index = tl.program_id(0)
    query_tile_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index\
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(nq, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    qi = tl.load(Q_block_ptr)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(nk, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(nk, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )

    o = tl.zeros((Bq, D), dtype=tl.float32)
    l = tl.zeros((Bq,), dtype=tl.float32)
    m = tl.full((Bq,), float('-inf'), dtype=tl.float32)

    scale = D ** -0.5
    mask_q = query_tile_index * Bq + tl.arange(0, Bq)
    mask_k_base = tl.arange(0, Bk)

    for _ in range(tl.cdiv(nk, Bk)):
        kj = tl.load(K_block_ptr)
        vj = tl.load(V_block_ptr)

        s = tl.dot(qi, kj.T) * scale
        if is_causal:
            mask_k = mask_k_base + _ * Bk
            s = tl.where(mask_q[:, None] >=  mask_k[None, :], s, -1e6)
        prev_m = m
        m = tl.maximum(prev_m, tl.max(s, axis=1))
        p = tl.exp(s - m[:, None])
        delta = tl.exp(prev_m - m)
        l = l * delta + tl.sum(p, axis=1)
        o = o * delta[:, None]
        o = tl.dot(p.to(dtype=V_block_ptr.dtype.element_ty), vj, acc=o)

        K_block_ptr = K_block_ptr.advance((Bk, 0))
        V_block_ptr = V_block_ptr.advance((Bk, 0))

    o = o * (1.0 / l)[:, None]
    l = tl.log(l) + m

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(nq, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, o.to(O_block_ptr.dtype.element_ty), boundary_check=(0,1))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(nq,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Bq,),
        block_shape=(Bq,),
        order=(0,),
    )
    tl.store(L_block_ptr, l.to(L_block_ptr.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=autotune_get_configs(['Bq', 'Bk']),
    key=['nq', 'nk', 'D']
)
@triton.jit
def flash_bwd_kernel_pass1(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    nq: tl.constexpr,
    nk: tl.constexpr,
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
        strides=(stride_doq, stride_dod),
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

    mask_k = key_tile_index * Bk + tl.arange(0, Bk)
    mask_q_base = tl.arange(0, Bq)
    scale = tl.full([], D ** -0.5, tl.float32)
    dK_sum = tl.zeros((Bk, D), dtype=tl.float32)
    dV_sum = tl.zeros((Bk, D), dtype=tl.float32)

    for _ in range(tl.cdiv(nq, Bq)):
        qi =  tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        s = tl.dot(qi, kj.trans()) * scale
        if is_causal:
            mask_q = mask_q_base + _ * Bq
            s = tl.where(mask_q[:, None] >=  mask_k[None, :], s, -1e6)

        li =  tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        p = tl.exp(s - li[:, None])
        
        doi = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero")
        p = p.to(Q_block_ptr.dtype.element_ty)
        dV_sum = tl.dot(p.trans(), doi, acc=dV_sum)
        dP = tl.dot(doi, vj.trans())

        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        ds = p * (dP - Di[:, None]) * scale
        ds = ds.to(Q_block_ptr.dtype.element_ty)
        dK_sum = tl.dot(ds.trans(), qi, acc=dK_sum)

        Q_block_ptr = Q_block_ptr.advance((Bq, 0))
        dO_block_ptr = dO_block_ptr.advance((Bq, 0))
        L_block_ptr = L_block_ptr.advance((Bq,))
        D_block_ptr = D_block_ptr.advance((Bq,))

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


@triton.autotune(
    configs=autotune_get_configs(['Bq', 'Bk']),
    key=['nq', 'nk', 'D']
)
@triton.jit
def flash_bwd_kernel_pass2(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    nq: tl.constexpr,
    nk: tl.constexpr,
    D: tl.constexpr,
    Bq: tl.constexpr, Bk: tl.constexpr,
    is_causal: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_tile_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(nq, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    qi =  tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(nq, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    doi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(nq,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Bq,),
        block_shape=(Bq,),
        order=(0,),
    )
    li =  tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(nq,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Bq,),
        block_shape=(Bq,),
        order=(0,),
    )
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

    
    mask_q = query_tile_index * Bq + tl.arange(0, Bq)
    mask_k_base = tl.arange(0, Bk)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(nk, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(nk, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )

    scale = tl.full([], D ** -0.5, tl.float32)
    dQ_sum = tl.zeros((Bq, D), dtype=tl.float32)

    for _ in range(tl.cdiv(nk, Bk)):
        kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s = tl.dot(qi, kj.trans()) * scale
        if is_causal:
            mask_k = mask_k_base + _ * Bk
            s = tl.where(mask_q[:, None] >=  mask_k[None, :], s, -1e6)

        p = tl.exp(s - li[:, None])
        dP = tl.dot(doi, vj.trans())
        ds = p * (dP - Di[:, None]) * scale
        ds = ds.to(Q_block_ptr.dtype.element_ty)
        dQ_sum = tl.dot(ds, kj, acc=dQ_sum)

        K_block_ptr = K_block_ptr.advance((Bk, 0))
        V_block_ptr = V_block_ptr.advance((Bk, 0))

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(nq, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    tl.store(dQ_block_ptr, dQ_sum.to(dQ_block_ptr.dtype.element_ty), boundary_check=(0,1))


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
        
        bs, nq, D = Q.shape
        bs, nk, D = K.shape

        O = torch.empty_like(Q)
        L = torch.empty((bs, nq,), dtype=Q.dtype, device=Q.device)

        def grid(META):
            return (bs, triton.cdiv(nq, META["Bq"]))
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            nq=nq, nk=nk,
            D=D,  
            is_causal=is_causal,
        )

        L = L.view(Q_shape[:-1])
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.Q_shape = Q_shape
        ctx.K_shape = K_shape
        ctx.is_causal = is_causal

        O = O.view(Q_shape)
        return O
    
    @staticmethod
    def backward_naive(ctx, dO):
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
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        # recover all parameters
        bs, nq, d = Q.shape
        nk = K.shape[1]
        is_causal = ctx.is_causal

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        D = torch.sum(O * dO, dim=-1)
        def grid_k(META):
            return (bs, triton.cdiv(nk, META["Bk"]))
        flash_bwd_kernel_pass1[grid_k](
            Q, K, V,
            dO, L, D,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            nq=nq, nk=nk,
            D=d,
            is_causal=is_causal,
        )
        def grid_q(META):
            return (bs, triton.cdiv(nq, META["Bq"]))
        flash_bwd_kernel_pass2[grid_q](
            Q, K, V,
            dO, L, D,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            nq=nq, nk=nk,
            D=d,
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
        Bq, Bk = Grid_Bq, Grid_Bk
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