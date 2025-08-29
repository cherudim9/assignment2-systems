import math
import torch
from einops import rearrange

Bq = 16
Bk = 16

class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_shape = Q.shape
        K_shape = K.shape
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
        mask = (torch.arange(nq)[:, None] >= torch.arange(nk)[None, :]).to(Q.device)

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
                    if is_causal:
                        s = torch.where(
                            mask[i * Bq : min((i + 1) * Bq, nq), j * Bk : min((j + 1) * Bk, nk)],
                            s, -1e6)

                    mi_jm1 = mi.clone()
                    mi = torch.maximum(mi_jm1, torch.max(s, dim=1).values)
                    pi = torch.exp(s - mi.unsqueeze(1))
                    li = li * torch.exp(mi_jm1 - mi) + torch.sum(pi, dim=1)
                    oi = torch.diag(torch.exp(mi_jm1 - mi)) @ oi + pi @ vj

                O[batch_idx, i * Bq : min((i + 1) * Bq, nq), :] = torch.diag(1.0 / li) @ oi
                L[batch_idx, i * Bq : min((i + 1) * Bq, nq)] = torch.log(li) + mi

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
