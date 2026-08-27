import torch


def fg2_step(S_old, k_tilde, v_tilde, alpha):
    F = torch.diag_embed(alpha) @ S_old
    prediction = k_tilde.unsqueeze(-2) @ F
    residual = v_tilde - prediction.squeeze(-2)
    correction = k_tilde.unsqueeze(-1) @ residual.unsqueeze(-2)
    S_new = F + correction
    return S_new


def fg2_ab(k_tilde, v_tilde, alpha):
    I = torch.eye(k_tilde.size(-1), device=k_tilde.device, dtype=k_tilde.dtype)
    A = (I - k_tilde.unsqueeze(-1) @ k_tilde.unsqueeze(-2)) @ torch.diag_embed(alpha)
    B = k_tilde.unsqueeze(-1) @ v_tilde.unsqueeze(-2)
    return A, B


def fg2_compact_step(S_old, k_tilde, v_tilde, alpha):
    A, B = fg2_ab(k_tilde, v_tilde, alpha)
    return A @ S_old + B


def fg2_scan_sequential(S0, K_tilde, V_tilde, Alpha):
    S = S0
    T = K_tilde.size(0)
    for t in range(T):
        k_tilde = K_tilde[t]
        v_tilde = V_tilde[t]
        alpha = Alpha[t]
        S = fg2_compact_step(S, k_tilde, v_tilde, alpha)
    return S
