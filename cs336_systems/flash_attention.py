import math
import torch
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
        ctx.save_for_backward(Q, K, V, O, L)

        O = O.view(Q_shape)
        return O
    
    @staticmethod
    def backward(ctx, do):
        Q, K, V, O, L = ctx.saved_tensors
