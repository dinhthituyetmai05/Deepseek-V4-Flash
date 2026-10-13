# QCAU = Query-Core Attention Unit.
#
# Pham vi cua file nay chi gom phan QCAU:
#   1. Nhan latent query c_q va dung W_UQ de mo rong thanh cac query head.
#   2. RMSNorm theo tung head va ap dung RoPE cho query.
#   3. Doc cac hang KV theo topk_idxs, sau do tinh sparse MQA.
#   4. Online softmax theo tile, co attention sink.
#   5. Ap dung inverse RoPE cho output va tra ve tensor cho GOPU.
#
# Cac phan nam ngoai QCAU khong duoc cai dat o day:
#   - ACU tao/chuan bi cac hang kv.
#   - LIU tao c_q va topk_idxs.
#   - GOPU nhan output o de lam output projection tiep theo.
#
# Ky hieu shape duoc dung trong comment:
#   B  = batch size
#   S  = so token/query trong mot lan goi
#   D_C = chieu latent query
#   H  = so query head cuc bo
#   Dh = so phan tu trong moi head
#   N  = so hang KV co san
#   K  = so dia chi KV ma moi query duoc phep doc

from typing import Callable, Optional, Union

import torch


# W_UQ co the duoc truyen vao duoi dang:
#   - torch.Tensor: ma tran/trong so query-up.
#   - callable: mot projector tu viet, nhan c_q va tra ve query.
WeightOrProjector = Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply the QCAU query-up weight: q = c_Q @ W_UQ.T."""
    # Chuc nang:
    # Ham phu nay thuc hien phep chieu tu latent query c_q len query day du.
    # No tuong duong linear khong bias: output = x @ weight.T.
    # Trong QCAU, ham nay duoc dung khi W_UQ la tensor 2D.
    # x co shape [B, S, D_C].
    # weight co shape [H*Dh, D_C], vi vay ket qua co shape
    # [B, S, H*Dh]. torch.matmul tu dong xu ly hai chieu dau B, S.
    if weight.ndim != 2:
        raise ValueError("2D W_UQ is required for the linear projection")
    return torch.matmul(x, weight.transpose(-1, -2))


def _rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """RMSNorm used after Query Up, with an optional per-head gamma."""
    # Chuc nang:
    # Chuan hoa vector query cua tung head theo RMSNorm.
    # Muc dich la dua moi query head ve cung thang do truoc khi tinh attention.
    # Neu co weight/gamma, ham nhan them he so hoc duoc sau khi chuan hoa.
    # Luon tinh RMSNorm bang float32 de giam sai so tich luy.
    # dtype goc duoc giu lai de tra ket qua ve dung dtype cua query.
    dtype = x.dtype
    x_float = x.float()

    # RMS(x) = sqrt(mean(x^2) + eps), tinh tren chieu cuoi cung Dh.
    # keepdim=True de he so RMS co the broadcast lai vao x.
    rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    y = x_float * rms

    # weight la gamma tuy chon cua RMSNorm.
    # Neu co, weight co shape [H, Dh] va broadcast theo [B, S, H, Dh].
    if weight is not None:
        y = y * weight.float()
    return y.to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    start_pos: int = 0,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE to the final rope_head_dim values of q or o."""
    # Chuc nang:
    # Ap Rotary Position Embedding vao phan rope dims cua query/output.
    # Khi inverse=False, day la phep gan thong tin vi tri cho query truoc attention.
    # Khi inverse=True, day la phep quay nguoc output sau attention.
    # RoPE chi tac dong len rope_head_dim phan tu cuoi moi head.
    # Khi inverse=True, dung tan so lien hop; day la phep quay nguoc
    # duoc ap dung cho output sau attention.
    if rope_head_dim == 0:
        return x
    if rope_head_dim % 2 != 0:
        raise ValueError("rope_head_dim must be even")
    if x.ndim != 4:
        raise ValueError("RoPE input must have shape [B,S,H,D]")
    if x.size(-1) < rope_head_dim:
        raise ValueError("last dimension is smaller than rope_head_dim")
    if freqs_cis.ndim != 2:
        raise ValueError("freqs_cis must have shape [S, rope_head_dim // 2]")

    # Hai gia tri thuc lien tiep tao thanh mot so phuc:
    # [real_0, imag_0, real_1, imag_1, ...].
    n_pairs = rope_head_dim // 2
    if freqs_cis.size(-1) != n_pairs:
        raise ValueError("freqs_cis has the wrong number of rotary pairs")

    # Co hai cach truyen bang tan so:
    #   1. Bang da duoc cat san, co dung S dong.
    #   2. Bang day du, trong do start_pos chon doan tuong ung voi query.
    if freqs_cis.size(0) == x.size(1):
        position_freqs = freqs_cis
    elif freqs_cis.size(0) >= start_pos + x.size(1):
        position_freqs = freqs_cis[start_pos:start_pos + x.size(1)]
    else:
        raise ValueError("freqs_cis does not cover the QCAU query positions")

    # Clone de khong sua tensor dau vao tai chinh cho phep autograd.
    out = x.clone()
    tail = out[..., -rope_head_dim:]

    # view_as_complex yeu cau cap cuoi cung co dang [..., 2].
    # contiguous() can thiet vi tail co the la view khong lien tuc.
    tail_complex = torch.view_as_complex(
        tail.float().contiguous().unflatten(-1, (n_pairs, 2))
    )

    # Frequences duoc reshape thanh [1, S, 1, n_pairs] de
    # broadcast cho moi batch va moi query head.
    freqs = position_freqs.conj() if inverse else position_freqs
    freqs = freqs.view(1, x.size(1), 1, n_pairs)

    # Nhan so phuc la phep quay 2D cua tung cap [real, imag].
    rotated = torch.view_as_real(tail_complex * freqs).flatten(-2)
    out[..., -rope_head_dim:] = rotated.to(out.dtype)
    return out


def _query_up(
    c_q: torch.Tensor,
    w_uq: WeightOrProjector,
    *,
    n_local_heads: Optional[int],
    head_dim: int,
    q_head_norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    start_pos: int,
    norm_eps: float,
) -> torch.Tensor:
    """QCAU Query Up, per-head RMSNorm and forward RoPE."""
    # Chuc nang:
    # Gom 3 buoc dau cua QCAU tren query:
    #   1. Query Up: bien c_q latent thanh query theo tung head.
    #   2. Per-head RMSNorm: chuan hoa moi query head.
    #   3. Forward RoPE: gan thong tin vi tri vao rope dims cua query.
    # Ket qua tra ve la q shape [B, S, H, Dh] da san sang de tinh QK.
    # Latent query phai co dang [B, S, D_C].
    if c_q.ndim != 3:
        raise ValueError("c_q must have shape [B,S,D_C]")

    # Ho tro ba dang Query Up:
    #   - callable: projector tu viet.
    #   - weight 2D: [H*Dh, D_C], dung phep linear thong thuong.
    #   - weight 3D: tao truc tiep tung head, voi mot trong hai layout.
    if callable(w_uq):
        q_projected = w_uq(c_q)
    elif w_uq.ndim == 2:
        q_projected = _linear(c_q, w_uq)
    elif w_uq.ndim == 3:
        # Layout [D_C, H, Dh]:
        # moi output head h va lane d la tong tren D_C.
        if w_uq.size(0) == c_q.size(-1):
            q_projected = torch.einsum("bsd,dhc->bshc", c_q, w_uq)

        # Layout [H, D_C, Dh], thuong tien cho weight phan theo head.
        elif w_uq.size(1) == c_q.size(-1):
            q_projected = torch.einsum("bsd,hdc->bshc", c_q, w_uq)
        else:
            raise ValueError("3D W_UQ must be [D_C,H,Dh] or [H,D_C,Dh]")
    else:
        raise ValueError("W_UQ must be a tensor or callable projector")

    # Chuan hoa output Query Up ve dang 4D [B, S, H, Dh].
    # Truong hop 3D [B, S, H*Dh], so head duoc suy ra tu head_dim.
    if q_projected.ndim == 3:
        if q_projected.size(-1) % head_dim != 0:
            raise ValueError("W_UQ output is not divisible by head_dim")
        inferred_heads = q_projected.size(-1) // head_dim
        if n_local_heads is not None and inferred_heads != n_local_heads:
            raise ValueError("W_UQ output does not match n_local_heads")
        n_local_heads = inferred_heads
        q = q_projected.unflatten(-1, (n_local_heads, head_dim))
    elif q_projected.ndim == 4:
        if q_projected.size(-1) != head_dim:
            raise ValueError("W_UQ output head dimension does not match head_dim")
        if n_local_heads is not None and q_projected.size(-2) != n_local_heads:
            raise ValueError("W_UQ output does not match n_local_heads")
        q = q_projected
    else:
        raise ValueError("W_UQ must produce [B,S,H*Dh] or [B,S,H,Dh]")

    # Neu co gamma theo head, shape phai khop [H, Dh].
    if q_head_norm_weight is not None:
        expected = (q.size(-2), q.size(-1))
        if tuple(q_head_norm_weight.shape) != expected:
            raise ValueError(f"q_head_norm_weight must have shape {expected}")

    # Thu tu chuan bi query:
    #   Query Up -> RMSNorm theo Dh -> RoPE thuan.
    q = _rms_norm(q, q_head_norm_weight, norm_eps)
    return _apply_rotary_emb(q, freqs_cis, rope_head_dim, start_pos)


def _sparse_attention_online(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    tile_size: int,
) -> torch.Tensor:
    """QCAU core MQA: tiled QK, online softmax, sink and weighted V."""
    # Chuc nang:
    # Day la loi attention cua QCAU.
    # Ham chi doc cac hang KV co dia chi trong topk_idxs, nen day la sparse attention.
    # Vi kv co shape [B, N, Dh] va duoc dung chung cho moi head, day la MQA.
    # Softmax duoc tinh online theo tile de khong can tao ma tran score lon trong bo nho.
    # attn_sink duoc dua vao mau so softmax nhu mot "sink token" khong co value.
    # q  : [B, S, H, Dh] - moi query head co mot vector rieng.
    # kv : [B, N, Dh]    - MQA dung cung mot KV vector cho moi query head.
    # topk_idxs: [B, S, K] - dia chi cac hang KV duoc phep truy cap.
    if q.ndim != 4:
        raise ValueError("q must have shape [B,S,H,Dh]")
    if kv.ndim != 3:
        raise ValueError("kv must have shape [B,N,Dh]")
    if topk_idxs.ndim != 3:
        raise ValueError("topk_idxs must have shape [B,S,K]")
    if kv.size(1) == 0:
        raise ValueError("kv must contain at least one latent row")
    if q.size(0) != kv.size(0) or q.size(0) != topk_idxs.size(0):
        raise ValueError("q, kv and topk_idxs batch dimensions must match")
    if q.size(1) != topk_idxs.size(1):
        raise ValueError("q and topk_idxs sequence dimensions must match")
    if q.size(-1) != kv.size(-1):
        raise ValueError("q and kv head dimensions must match")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    bsz, seqlen, n_heads, head_dim = q.shape
    if attn_sink.numel() != n_heads:
        raise ValueError("attn_sink must have one value per local query head")

    # Tinh toan attention va toan bo trang thai online bang float32.
    # Ket qua cuoi cung moi duoc cast ve dtype cua q.
    q_float = q.float()
    kv_float = kv.float()
    idxs = topk_idxs.to(device=q.device, dtype=torch.long)

    # Trang thai cua online softmax:
    #   acc_o     = tong co trong so cua cac vector V
    #   sum_exp   = tong cac exp(score) da duoc chuan hoa
    #   scores_max = score lon nhat da gap, dung de tranh overflow
    acc_o = torch.zeros(
        bsz, seqlen, n_heads, head_dim,
        device=q.device,
        dtype=torch.float32,
    )
    sum_exp = torch.zeros(
        bsz, seqlen, n_heads,
        device=q.device,
        dtype=torch.float32,
    )
    scores_max = torch.full(
        (bsz, seqlen, n_heads),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    # Khong tao ma tran attention day du [B,S,H,K] trong mot lan.
    # Thay vao do, duyet K theo cac tile nho de tiet kiem bo nho.
    for tile_base in range(0, idxs.size(-1), tile_size):
        tile_idxs = idxs[..., tile_base:tile_base + tile_size]

        # Chi so hop le nam trong [0, N).
        # -1 thuong dung de danh dau vi tri khong co KV/future token.
        valid = (tile_idxs >= 0) & (tile_idxs < kv.size(1))

        # Gather khong chap nhan index am/out-of-range, nen clamp de
        # doc an toan. Mask valid o cac buoc sau se loai dong gia nay.
        safe_idxs = tile_idxs.clamp(0, kv.size(1) - 1)

        # Mo rong moi chi so thanh Dh lan de gather ca mot vector KV.
        gather_index = safe_idxs.unsqueeze(-1).expand(*safe_idxs.shape, head_dim)
        kv_tile = torch.gather(
            kv_float,
            1,
            gather_index.reshape(bsz, -1, head_dim),
        ).view(bsz, seqlen, -1, head_dim)

        # Cac dia chi khong hop le khong duoc dong gop vao gia tri V.
        kv_tile = torch.where(
            valid.unsqueeze(-1),
            kv_tile,
            torch.zeros_like(kv_tile),
        )

        # QK dot product:
        #   q          [B,S,H,Dh]
        #   kv_tile    [B,S,T,Dh]
        #   scores     [B,S,H,T]
        # Cung mot kv_tile duoc dung lai cho tat ca H query head: MQA.
        scores = torch.einsum("bshd,bstd->bsht", q_float, kv_tile)
        scores = scores * softmax_scale

        # Dat score khong hop le thanh -inf de softmax bo qua.
        scores = torch.where(
            valid.unsqueeze(2),
            scores,
            torch.full_like(scores, float("-inf")),
        )

        # Cap nhat max theo online softmax.
        # tile_has_valid giup tranh phep -inf - (-inf) tao NaN
        # khi mot tile chi chua cac dia chi khong hop le.
        previous_max = scores_max
        tile_has_valid = valid.any(dim=-1).unsqueeze(-1)
        tile_max = scores.amax(dim=-1)
        next_max = torch.maximum(previous_max, tile_max)
        scores_max = torch.where(tile_has_valid, next_max, previous_max)

        # Khi max thay doi, rescale trang thai cua cac tile truoc.
        # Neu tile rong, giu old_scale = 1.
        old_scale = torch.where(
            tile_has_valid,
            torch.exp(previous_max - scores_max),
            torch.ones_like(scores_max),
        )

        # Khi scores_max van la -inf, dung 0 tam thoi de exp khong tao NaN.
        safe_max = torch.where(
            torch.isfinite(scores_max),
            scores_max,
            torch.zeros_like(scores_max),
        )

        # Xac suat chua chuan hoa cua tile hien tai.
        probs = torch.exp(scores - safe_max.unsqueeze(-1))
        probs = torch.where(
            valid.unsqueeze(2),
            probs,
            torch.zeros_like(probs),
        )

        # Cong don online:
        #   l_new = l_old * old_scale + sum(probs)
        #   o_new = o_old * old_scale + sum(probs * V)
        sum_exp = sum_exp * old_scale + probs.sum(dim=-1)
        acc_o = (
            acc_o * old_scale.unsqueeze(-1)
            + torch.einsum("bsht,bstd->bshd", probs, kv_tile)
        )

    # Attention sink khong co vector V tuong ung.
    # No chi them mot hang vao mau so softmax, khong thay doi acc_o.
    sink = attn_sink.to(device=q.device, dtype=torch.float32).view(1, 1, n_heads)
    finite_scores = torch.isfinite(scores_max)
    finite_sink = torch.isfinite(sink)
    sink_term = torch.where(
        finite_scores,
        torch.exp(sink - scores_max),
        finite_sink.to(sum_exp.dtype),
    )
    sum_exp = sum_exp + sink_term

    # Chia tong co trong so de nhan output attention.
    # clamp_min ngan chia cho 0 trong truong hop du lieu dac biet.
    return (
        acc_o
        / sum_exp.clamp_min(torch.finfo(sum_exp.dtype).tiny).unsqueeze(-1)
    ).to(q.dtype)


def QCAU(
    *,
    c_q: torch.Tensor,
    w_uq: WeightOrProjector,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_head_norm_weight: Optional[torch.Tensor] = None,
    n_local_heads: Optional[int] = None,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    start_pos: int = 0,
    norm_eps: float = 1e-6,
    softmax_scale: Optional[float] = None,
    tile_size: int = 64,
) -> torch.Tensor:
    """Run exactly the Query-Core Attention Unit.

    Inputs are produced by the preceding ACU/LIU stages:
      c_q:       [B,S,D_C] latent query from Query Down
      w_uq:      Query Up weight/projector
      kv:        [B,N,Dh] selected compressed KV plus sliding-window KV
      topk_idxs: [B,S,K] addresses into kv; invalid entries are -1
      attn_sink: [H_local] attention-sink scalar for each query head
      freqs_cis: [S, rope_head_dim // 2] table, or a longer table with
                  start_pos selecting the QCAU query positions

    The output is `o` with shape [B,S,H_local,Dh]. GOPU is downstream.
    """
    # Chuc nang:
    # Day la ham public cua file, dai dien cho toan bo khoi QCAU.
    # No nhan c_q, W_UQ, KV va topk_idxs tu cac khoi ben ngoai,
    # sau do chay dung phan QCAU:
    #   Query Up -> RMSNorm -> RoPE -> sparse MQA -> online softmax/sink -> inverse RoPE.
    # Ham nay khong tao c_q, khong tao/chon KV, va khong lam output projection GOPU.
    # Kiem tra cac shape o bien gioi public cua QCAU.
    # Cac loi shape duoc bao som de khong bi loi kho hieu trong einsum/gather.
    if c_q.ndim != 3:
        raise ValueError("c_q must have shape [B,S,D_C]")
    if kv.ndim != 3:
        raise ValueError("kv must have shape [B,N,Dh]")
    if topk_idxs.ndim != 3:
        raise ValueError("topk_idxs must have shape [B,S,K]")
    if c_q.size(0) != kv.size(0) or c_q.size(0) != topk_idxs.size(0):
        raise ValueError("c_q, kv and topk_idxs batch dimensions must match")
    if c_q.size(1) != topk_idxs.size(1):
        raise ValueError("c_q and topk_idxs sequence dimensions must match")

    # Buoc 1: dung latent query de tao query day du theo head,
    # RMSNorm va RoPE thuan.
    q = _query_up(
        c_q,
        w_uq,
        n_local_heads=n_local_heads,
        head_dim=head_dim,
        q_head_norm_weight=q_head_norm_weight,
        freqs_cis=freqs_cis,
        rope_head_dim=rope_head_dim,
        start_pos=start_pos,
        norm_eps=norm_eps,
    )

    # Q phai co cung chieu Dh voi moi hang KV.
    if q.size(-1) != kv.size(-1):
        raise ValueError("Query Up head dimension must match KV width")
    if attn_sink.numel() != q.size(-2):
        raise ValueError("attn_sink must match the number of local query heads")

    # Neu nguoi dung khong truyen scale, dung scale attention tieu chuan
    # 1/sqrt(Dh).
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5

    # Buoc 2: attention sparse MQA theo tile, online softmax va sink.
    o = _sparse_attention_online(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale,
        tile_size,
    )

    # Buoc 3: quay nguoc phan RoPE cua output.
    # Tensor tra ve van co dang [B, S, H_local, Dh] de GOPU xu ly tiep.
    return _apply_rotary_emb(
        o,
        freqs_cis,
        rope_head_dim,
        start_pos,
        inverse=True,
    )
