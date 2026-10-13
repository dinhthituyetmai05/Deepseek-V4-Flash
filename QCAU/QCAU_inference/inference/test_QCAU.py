import torch

from dinhanh.QCAU.inference.QCAU import QCAU, _sparse_attention_online


def _dense_attention_reference(q, kv, topk_idxs, attn_sink, scale):
    outputs = []
    for b in range(q.size(0)):
        rows = []
        for s in range(q.size(1)):
            valid = (topk_idxs[b, s] >= 0) & (topk_idxs[b, s] < kv.size(1))
            selected = kv[b, topk_idxs[b, s].clamp(0, kv.size(1) - 1)]
            head_outputs = []
            for h in range(q.size(2)):
                logits = torch.einsum("d,kd->k", q[b, s, h], selected) * scale
                logits = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
                probs = torch.softmax(torch.cat([logits, attn_sink[h:h + 1]]), dim=0)[:-1]
                head_outputs.append(torch.einsum("k,kd->d", probs, selected))
            rows.append(torch.stack(head_outputs))
        outputs.append(torch.stack(rows))
    return torch.stack(outputs)


def test_online_attention_matches_dense_reference():
    torch.manual_seed(1)
    q = torch.randn(2, 3, 4, 8)
    kv = torch.randn(2, 11, 8)
    topk_idxs = torch.tensor(
        [
            [[0, 3, 5, -1, 10, 10, -1], [1, 2, 8, 9, -1, -1, -1], [4, 6, 7, 10, 0, -1, -1]],
            [[2, 4, 6, 8, 10, -1, -1], [0, 1, 2, 3, 4, 5, 6], [9, 10, -1, -1, -1, -1, -1]],
        ],
        dtype=torch.long,
    )
    sink = torch.randn(4)
    scale = 8 ** -0.5

    actual = _sparse_attention_online(q, kv, sink, topk_idxs, scale, tile_size=3)
    expected = _dense_attention_reference(q, kv, topk_idxs, sink, scale).to(q.dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_qcau_query_up_core_and_inverse_rope():
    torch.manual_seed(2)
    bsz, seqlen, q_rank, n_heads, head_dim, kv_len = 2, 3, 5, 3, 4, 9
    c_q = torch.randn(bsz, seqlen, q_rank)
    w_uq = torch.randn(n_heads * head_dim, q_rank)
    q_head_norm = torch.randn(n_heads, head_dim).abs() + 0.5
    kv = torch.randn(bsz, kv_len, head_dim)
    topk_idxs = torch.tensor(
        [
            [[0, 1, 4, 7, -1, -1], [0, 2, 3, 8, -1, -1], [1, 2, 5, 6, 7, -1]],
            [[0, 1, 2, 4, 6, -1], [1, 3, 5, 7, -1, -1], [0, 2, 4, 8, -1, -1]],
        ],
        dtype=torch.long,
    )
    freqs = torch.polar(
        torch.ones(seqlen, head_dim // 2),
        torch.randn(seqlen, head_dim // 2),
    )
    sink = torch.randn(n_heads)

    actual = QCAU(
        c_q=c_q,
        w_uq=w_uq,
        q_head_norm_weight=q_head_norm,
        kv=kv,
        topk_idxs=topk_idxs,
        attn_sink=sink,
        freqs_cis=freqs,
        n_local_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=head_dim,
        tile_size=2,
    )

    q = torch.matmul(c_q, w_uq.transpose(-1, -2)).unflatten(-1, (n_heads, head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    q = q * q_head_norm
    q_complex = torch.view_as_complex(q.float().unflatten(-1, (-1, 2)))
    q = torch.view_as_real(q_complex * freqs.view(1, seqlen, 1, -1)).flatten(-2).to(q.dtype)
    expected = _sparse_attention_online(q, kv, sink, topk_idxs, head_dim ** -0.5, tile_size=2)
    out_complex = torch.view_as_complex(expected.float().unflatten(-1, (-1, 2)))
    expected = torch.view_as_real(
        out_complex * freqs.conj().view(1, seqlen, 1, -1)
    ).flatten(-2).to(expected.dtype)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_qcau_accepts_head_major_query_up_weight():
    torch.manual_seed(3)
    c_q = torch.randn(1, 2, 4)
    w_uq = torch.randn(3, 4, 5)
    kv = torch.randn(1, 4, 5)
    idx = torch.tensor([[[0, 1], [1, 2]]])
    freqs = torch.polar(torch.ones(2, 2), torch.randn(2, 2))

    output = QCAU(
        c_q=c_q,
        w_uq=w_uq,
        kv=kv,
        topk_idxs=idx,
        attn_sink=torch.zeros(3),
        freqs_cis=freqs,
        n_local_heads=3,
        head_dim=5,
        rope_head_dim=4,
    )

    assert output.shape == (1, 2, 3, 5)
