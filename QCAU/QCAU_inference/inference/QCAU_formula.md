# Cong thuc toan hoc cua khoi QCAU

Tai lieu nay mo ta cong thuc cua ham `QCAU()` trong `QCAU.py`.
Boundary theo code hien tai:

$$
c_q,\; W_{UQ}
\rightarrow \text{QueryUp}
\rightarrow \text{RMSNorm}
\rightarrow \text{RoPE}
\rightarrow \text{Sparse MQA + Sink}
\rightarrow \text{RoPE}^{-1}
\rightarrow o
$$

## 1. Ky hieu va shape

| Ky hieu | Y nghia |
|---|---|
| $B$ | batch size |
| $S$ | so query token trong lan goi hien tai |
| $D_C$ | chieu latent query cua $c_q$ |
| $H$ | so local query heads |
| $D_h$ | head dimension |
| $D_r$ | rope head dimension |
| $K$ | so dia chi KV tren moi query |
| $T$ | tile size |
| $N$ | so hang KV co san |

Input:

$$
c_q \in \mathbb{R}^{B \times S \times D_C}
$$

$$
W_{UQ} \in \mathbb{R}^{(H D_h) \times D_C}
$$

$$
kv \in \mathbb{R}^{B \times N \times D_h}
$$

$$
topk\_idxs \in \mathbb{Z}^{B \times S \times K}
$$

$$
attn\_sink \in \mathbb{R}^{H}
$$

$$
freqs\_cis \in \mathbb{C}^{L \times D_r/2}
$$

Trong code QCAU hien tai, `kv` duoc dung chung lam ca Key va Value:

$$
K_j = kv_j,\qquad V_j = kv_j
$$

Vi vay day la MQA style: moi head co query rieng, nhung dung chung KV.

## 2. Query Up

Voi `w_uq` dang 2D, code tinh:

$$
q^{proj}_{b,s,:}
= c_{q,b,s,:} W_{UQ}^{T}
$$

Shape:

$$
q^{proj} \in \mathbb{R}^{B \times S \times (H D_h)}
$$

Sau do reshape thanh query theo head:

$$
q^{0}_{b,s,h,d}
= \operatorname{reshape}(q^{proj})_{b,s,h,d}
$$

$$
q^{0} \in \mathbb{R}^{B \times S \times H \times D_h}
$$

Neu `w_uq` la 3D layout $[D_C,H,D_h]$:

$$
q^{0}_{b,s,h,d}
= \sum_{c=0}^{D_C-1}
c_{q,b,s,c} W_{UQ,c,h,d}
$$

Neu `w_uq` la 3D layout $[H,D_C,D_h]$:

$$
q^{0}_{b,s,h,d}
= \sum_{c=0}^{D_C-1}
c_{q,b,s,c} W_{UQ,h,c,d}
$$

## 3. Per-head RMSNorm

Voi moi vector query cua tung head:

$$
r_{b,s,h}
=
\sqrt{
  \frac{1}{D_h}
  \sum_{d=0}^{D_h-1}
  \left(q^{0}_{b,s,h,d}\right)^2
  + \epsilon
}
$$

Neu khong co `q_head_norm_weight`:

$$
q^{1}_{b,s,h,d}
=
\frac{q^{0}_{b,s,h,d}}{r_{b,s,h}}
$$

Neu co gamma rieng:

$$
q^{1}_{b,s,h,d}
=
\frac{q^{0}_{b,s,h,d}}{r_{b,s,h}}
\cdot \gamma_{h,d}
$$

Trong do:

$$
\gamma = q\_head\_norm\_weight
$$

Neu `q_head_norm_weight = None`, co the xem:

$$
\gamma_{h,d} = 1
$$

## 4. RoPE forward tren query

RoPE chi ap dung tren $D_r$ phan tu cuoi cua moi vector head. Dat:

$$
P = \frac{D_r}{2}
$$

Voi moi pair trong phan rope tail:

$$
z_{b,s,h,p}
=
q^{1}_{b,s,h,D_h-D_r+2p}
+
i q^{1}_{b,s,h,D_h-D_r+2p+1}
$$

Tan so vi tri:

$$
f_{s,p}
=
freqs\_cis_{start\_pos+s,p}
$$

Neu `freqs_cis` da co do dai dung bang $S$, code dung truc tiep:

$$
f_{s,p}
=
freqs\_cis_{s,p}
$$

RoPE forward:

$$
z'_{b,s,h,p}
=
z_{b,s,h,p} f_{s,p}
$$

Phan non-rope giu nguyen. Sau khi chuyen complex pairs ve real pairs:

$$
q \in \mathbb{R}^{B \times S \times H \times D_h}
$$

## 5. Tap KV hop le

Voi moi query $(b,s)$:

$$
I_{b,s,k}
=
topk\_idxs_{b,s,k}
$$

Index hop le:

$$
0 \le I_{b,s,k} < N
$$

Tap index hop le:

$$
\mathcal{I}_{b,s}
=
\{k \mid 0 \le I_{b,s,k} < N\}
$$

Voi $k \in \mathcal{I}_{b,s}$:

$$
j = I_{b,s,k}
$$

$$
K_{b,s,k,:} = kv_{b,j,:},
\qquad
V_{b,s,k,:} = kv_{b,j,:}
$$

Voi index invalid, code gan:

$$
score = -\infty,
\qquad
V = 0
$$

## 6. QK score

Voi $k \in \mathcal{I}_{b,s}$:

$$
x_{b,s,h,k}
=
\alpha
\sum_{d=0}^{D_h-1}
q_{b,s,h,d}
kv_{b,I_{b,s,k},d}
$$

Trong do:

$$
\alpha = softmax\_scale
$$

Neu `softmax_scale = None`, code dung:

$$
\alpha = \frac{1}{\sqrt{D_h}}
$$

Nen mac dinh:

$$
x_{b,s,h,k}
=
\frac{
q_{b,s,h}
\cdot
kv_{b,I_{b,s,k}}
}{
\sqrt{D_h}
}
$$

## 7. Sparse MQA dang gon

Neu bo qua chi tiet online tiling, output sparse attention truoc inverse RoPE la:

$$
o^{raw}_{b,s,h}
=
\frac{
  \sum\limits_{k \in \mathcal{I}_{b,s}}
  e^{x_{b,s,h,k}}
  kv_{b,I_{b,s,k}}
}{
  \sum\limits_{k \in \mathcal{I}_{b,s}}
  e^{x_{b,s,h,k}}
  +
  e^{attn\_sink_h}
}
$$

Attention sink chi nam trong mau so. No khong co vector value rieng, nen khong xuat hien trong tu so.

## 8. Online softmax theo tile

Code tinh cong thuc tren theo tung tile de tiet kiem bo nho va tranh overflow.
Chia tap index thanh cac tile:

$$
\mathcal{I}_{b,s}
=
\mathcal{T}_1
\cup
\mathcal{T}_2
\cup
\cdots
\cup
\mathcal{T}_R
$$

Khoi tao cho moi $(b,s,h)$:

$$
m_0 = -\infty
$$

$$
\ell_0 = 0
$$

$$
a_0 = 0 \in \mathbb{R}^{D_h}
$$

Trong do:

| Ky hieu | Tuong ung trong code |
|---|---|
| $m_r$ | `scores_max` |
| $\ell_r$ | `sum_exp` |
| $a_r$ | `acc_o` |

### 8.1. Tile max

Voi tile thu $r$:

$$
t_r
=
\max_{k \in \mathcal{T}_r}
x_{b,s,h,k}
$$

### 8.2. Running max

$$
m_r
=
\max(m_{r-1}, t_r)
$$

### 8.3. Scale lai tong cu

$$
\lambda_r
=
e^{m_{r-1}-m_r}
$$

### 8.4. Mau so online

$$
\ell_r
=
\ell_{r-1}\lambda_r
+
\sum_{k \in \mathcal{T}_r}
e^{x_{b,s,h,k}-m_r}
$$

Voi index invalid, phan dong gop bang 0.

### 8.5. Tu so / weighted V online

$$
a_r
=
a_{r-1}\lambda_r
+
\sum_{k \in \mathcal{T}_r}
e^{x_{b,s,h,k}-m_r}
kv_{b,I_{b,s,k}}
$$

Sau khi da quy tat ca ve cung moc max $m_R$, co the hinh dung:

$$
a_R
=
\sum_{k \in \mathcal{I}_{b,s}}
e^{x_{b,s,h,k}-m_R}
kv_{b,I_{b,s,k}}
$$

Day la phan "Weighted V sum" trong online softmax.

## 9. Attention sink trong online softmax

Sau tile cuoi:

$$
sink\_term_{h}
=
e^{attn\_sink_h - m_R}
$$

Mau so cuoi:

$$
\ell_{final}
=
\ell_R
+
e^{attn\_sink_h - m_R}
$$

Output truoc inverse RoPE:

$$
o^{raw}_{b,s,h}
=
\frac{a_R}{\ell_{final}}
$$

Viet day du:

$$
o^{raw}_{b,s,h}
=
\frac{
  \sum\limits_{k \in \mathcal{I}_{b,s}}
  e^{x_{b,s,h,k}-m_R}
  kv_{b,I_{b,s,k}}
}{
  \sum\limits_{k \in \mathcal{I}_{b,s}}
  e^{x_{b,s,h,k}-m_R}
  +
  e^{attn\_sink_h-m_R}
}
$$

Cong thuc nay tuong duong voi cong thuc softmax gon o tren, vi ca tu va mau deu duoc chia cho $e^{m_R}$.

## 10. RoPE inverse tren output

Sau sparse attention, QCAU ap dung inverse RoPE tren $D_r$ phan tu cuoi cua $o^{raw}$.

Chuyen tail thanh complex:

$$
u_{b,s,h,p}
=
o^{raw}_{b,s,h,D_h-D_r+2p}
+
i o^{raw}_{b,s,h,D_h-D_r+2p+1}
$$

Dung cung tan so:

$$
f_{s,p}
=
freqs\_cis_{start\_pos+s,p}
$$

RoPE inverse:

$$
u'_{b,s,h,p}
=
u_{b,s,h,p}
\cdot
\overline{f_{s,p}}
$$

Phan non-rope giu nguyen. Sau khi chuyen complex pairs ve real pairs:

$$
o \in \mathbb{R}^{B \times S \times H \times D_h}
$$

## 11. Cong thuc tong hop cua QCAU

Pipeline:

$$
q^{proj} = c_q W_{UQ}^{T}
$$

$$
q^0 = \operatorname{reshape}(q^{proj})
$$

$$
q^1 = \operatorname{RMSNorm}(q^0)
$$

$$
q = \operatorname{RoPE}(q^1, freqs\_cis)
$$

$$
o^{raw}
=
\operatorname{SparseMQA}
(q, kv, topk\_idxs, attn\_sink)
$$

$$
o
=
\operatorname{RoPE}^{-1}
(o^{raw}, freqs\_cis)
$$

Cong thuc day du:

$$
o_{b,s,h}
=
\operatorname{RoPE}^{-1}
\left(
  \frac{
    \sum\limits_{k \in \mathcal{I}_{b,s}}
    e^{x_{b,s,h,k}}
    kv_{b,I_{b,s,k}}
  }{
    \sum\limits_{k \in \mathcal{I}_{b,s}}
    e^{x_{b,s,h,k}}
    +
    e^{attn\_sink_h}
  },
  freqs\_cis
\right)
$$

voi:

$$
x_{b,s,h,k}
=
\alpha
q_{b,s,h}
\cdot
kv_{b,I_{b,s,k}}
$$

va:

$$
\alpha
=
\frac{1}{\sqrt{D_h}}
\quad \text{neu } softmax\_scale = None
$$

## 12. Bien nao la weight

Trong boundary cua `QCAU.py`:

| Bien | Co phai weight can luu? | Ghi chu |
|---|---|---|
| $W_{UQ}$ / `w_uq` | Co | Query Up projector |
| `attn_sink` | Co | Learned scalar moi head |
| `q_head_norm_weight` | Chi neu dung | Code hien tai co the de `None`, luc do gamma = 1 |
| `freqs_cis` | Khong | Bang RoPE sinh lai tu config |
| `kv` | Khong | Activation/cache, dung lam ca K va V |
| `topk_idxs` | Khong | Index runtime |
| `c_q` | Khong | Activation input |
| `q`, `o_raw`, `o` | Khong | Tensor trung gian/output |
| `acc_o`, `sum_exp`, `scores_max` | Khong | Bien runtime cua online softmax |
