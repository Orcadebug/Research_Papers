import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fg2_chunked import apply_chunk_summary, full_multi_chunk_scan, summarize_chunk
from fg2_reference import fg2_ab, fg2_compact_step, fg2_scan_sequential, fg2_step


def test_fg2_step_equivalence():
    B, d = 2, 4
    S_old = torch.randn(B, d, d)
    k_tilde = torch.randn(B, d)
    v_tilde = torch.randn(B, d)
    alpha = torch.sigmoid(torch.randn(B, d))

    step_out = fg2_step(S_old, k_tilde, v_tilde, alpha)
    compact_out = fg2_compact_step(S_old, k_tilde, v_tilde, alpha)

    print(torch.allclose(step_out, compact_out, atol=1e-5))


def test_fg2_scan_sequential():
    T, B, d = 3, 2, 4
    S0 = torch.randn(B, d, d)
    K_tilde = torch.randn(T, B, d)
    V_tilde = torch.randn(T, B, d)
    Alpha = torch.sigmoid(torch.randn(T, B, d))

    S_final = fg2_scan_sequential(S0, K_tilde, V_tilde, Alpha)
    print(S_final.shape)


def test_summarize_chunk():
    T, B, d = 3, 2, 4
    S0 = torch.randn(B, d, d)
    K_tilde = torch.randn(T, B, d)
    V_tilde = torch.randn(T, B, d)
    Alpha = torch.sigmoid(torch.randn(T, B, d))

    A_list = []
    B_list = []
    for t in range(T):
        A_t, B_t = fg2_ab(K_tilde[t], V_tilde[t], Alpha[t])
        A_list.append(A_t)
        B_list.append(B_t)

    A_chunk = torch.stack(A_list, dim=0)
    B_chunk = torch.stack(B_list, dim=0)

    P, C = summarize_chunk(A_chunk, B_chunk)

    sequential = fg2_scan_sequential(S0, K_tilde, V_tilde, Alpha)
    chunked = P @ S0 + C

    print(torch.allclose(sequential, chunked, atol=1e-5))


def test_apply_chunk_summary():
    T, B, d = 3, 2, 4
    S0 = torch.randn(B, d, d)
    K_tilde = torch.randn(T, B, d)
    V_tilde = torch.randn(T, B, d)
    Alpha = torch.sigmoid(torch.randn(T, B, d))

    sequential = fg2_scan_sequential(S0, K_tilde, V_tilde, Alpha)
    chunked = apply_chunk_summary(S0, K_tilde, V_tilde, Alpha)

    print(torch.allclose(sequential, chunked, atol=1e-5))


def test_full_multi_chunk_scan():
    T, B, d = 6, 2, 4
    S0 = torch.randn(B, d, d)
    K_tilde = torch.randn(T, B, d)
    V_tilde = torch.randn(T, B, d)
    Alpha = torch.sigmoid(torch.randn(T, B, d))
    chunk_size = 2

    sequential = fg2_scan_sequential(S0, K_tilde, V_tilde, Alpha)
    multi_chunk = full_multi_chunk_scan(S0, K_tilde, V_tilde, Alpha, chunk_size)

    print(torch.allclose(sequential, multi_chunk, atol=1e-5))


def main():
    print("FG² capstone tests")
    test_fg2_step_equivalence()
    test_fg2_scan_sequential()
    test_summarize_chunk()
    test_apply_chunk_summary()
    test_full_multi_chunk_scan()


if __name__ == "__main__":
    main()
