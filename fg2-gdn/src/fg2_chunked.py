import torch

from fg2_reference import fg2_ab


def summarize_chunk(A_chunk, B_chunk):
    _, B, d, _ = A_chunk.shape
    P = torch.eye(d, device=A_chunk.device, dtype=A_chunk.dtype).expand(B, d, d)
    C = torch.zeros_like(B_chunk[0])
    T = A_chunk.size(0)
    for t in range(T):
        P = A_chunk[t] @ P
        C = A_chunk[t] @ C + B_chunk[t]
    return P, C


def apply_chunk_summary(S_old, k_tilde, v_tilde, alpha):
    T = k_tilde.size(0)
    A_list = []
    B_list = []
    for t in range(T):
        A_t, B_t = fg2_ab(k_tilde[t], v_tilde[t], alpha[t])
        A_list.append(A_t)
        B_list.append(B_t)
    A_chunk = torch.stack(A_list, dim=0)
    B_chunk = torch.stack(B_list, dim=0)
    P, C = summarize_chunk(A_chunk, B_chunk)
    state = P @ S_old + C
    return state


def combine_PCchunks(P_left, C_left, P_right, C_right):
    P_combined = P_right @ P_left
    C_combined = P_right @ C_left + C_right
    return P_combined, C_combined


def full_multi_chunk_scan(S0, K_tilde, V_tilde, Alpha, chunk_size):
    T = K_tilde.size(0)
    A_list = []
    B_list = []
    for t in range(T):
        A_t, B_t = fg2_ab(K_tilde[t], V_tilde[t], Alpha[t])
        A_list.append(A_t)
        B_list.append(B_t)
    A_seq = torch.stack(A_list, dim=0)
    B_seq = torch.stack(B_list, dim=0)

    chunk_summaries = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        A_chunk = A_seq[start:end]
        B_chunk = B_seq[start:end]
        P_chunk, C_chunk = summarize_chunk(A_chunk, B_chunk)
        chunk_summaries.append((P_chunk, C_chunk))

    P, C = chunk_summaries[0]
    for P_right, C_right in chunk_summaries[1:]:
        P, C = combine_PCchunks(P, C, P_right, C_right)

    return P @ S0 + C
