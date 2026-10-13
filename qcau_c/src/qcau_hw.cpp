#include "../include/Core.h"
#define fp8 float


const int seq_len = 16, q_lora_rank = 1024, n_heads = 64, head_dim = 512, rope_head_dim = 64;
const int N = 20, K = 20, norm_eps = 1e-6;
const int block = 64;
constexpr int num_blocks = K / block + (K % block != 0 ? 1 : 0);

float WTB[q_lora_rank * n_heads * head_dim + (q_lora_rank * n_heads * head_dim >> 14) + seq_len * rope_head_dim + n_heads];
__bf16 RAB[seq_len * q_lora_rank + seq_len * n_heads * head_dim];
__bf16 KVB_kv[N * head_dim];
int KVB_topk[seq_len * K];

__bf16 PSA_qraw[seq_len][n_heads][head_dim];


template <typename T1, typename T2, int R, int C, int M>
void gemm(
    T1 X1[R][M], T1* X2, 
    bool transpose, bool accum, 
    T2 Y[R][C]
) {
    for (int i = 0; i < R; i++) 
    for (int j = 0; j < C; j++) {
        if (!accum) 
            Y[i][j] = 0;
        for (int m = 0; m < M; m++) {
            float B = transpose ? X2[j * M + m] : X2[m * C + j];
            Y[i][j] += (T2) (X1[i][m] * B);
        }
    }
}


void quant_gemm(
    fp8 X1[seq_len][q_lora_rank], 
    fp8 scale_1[seq_len][q_lora_rank >> 7]
) {
    constexpr int s2_row = q_lora_rank >> 7, s2_col = n_heads * head_dim >> 7;

    float* WTB_w_uq = &WTB[0];
    float* WTB_w_uq_scale = &WTB[q_lora_rank * n_heads * head_dim];

    float x1[seq_len][128];
    float x2[128][128];
    float scale_2[s2_col][s2_row];
    float y[seq_len][128];
    float acc[seq_len][128];

    // load scale 2
    for (int i = 0; i < s2_row; i++) 
    for (int j = 0; j < s2_col; j++) 
        scale_2[j][i] = WTB_w_uq_scale[j * s2_row + i];

    for (int c = 0; c < s2_col; c++) {
        for (int k = 0; k < s2_row; k++) {

            // load x1
            for (int i = 0; i < seq_len; i++) 
            for (int j = 0; j < 128; j++) 
                x1[i][j] = X1[i][(k << 7) + j];
            
            // load x2
            for (int i = 0; i < 128; i++) 
            for (int j = 0; j < 128; j++) 
                x2[j][i] = WTB_w_uq[((c << 7) + j) * q_lora_rank + (k << 7) + i];

            // cal y
            gemm<float, float, seq_len, 128, 128>(x1, (float*) x2, true, false, y);

            // acc += y
            for (int i = 0; i < seq_len; i++) 
            for (int j = 0; j < 128; j++) {
                float acc_num = y[i][j] * (scale_1[i][k] * scale_2[c][k]);
                if (k) 
                    acc[i][j] += acc_num;
                else 
                    acc[i][j] = acc_num;
            }
        }
        
        // store acc
        for (int i = 0; i < seq_len; i++) 
        for (int j = 0; j < 128; j++) {
            int id = (c << 7) + j;
            PSA_qraw[i][id >> 9][id & 0x1ff] = (__bf16) acc[i][j];
        }
    }
}

void rms_norm(
    __bf16 out[seq_len][n_heads][head_dim]
) {
    float _1_dim = 1.0f / head_dim;

    // local buffer
    float vec_x[head_dim];

    for (int s = 0; s < seq_len; s++) 
    for (int h = 0; h < n_heads; h++) {

        // load vec_x
        for (int c = 0; c < head_dim; c++) 
            vec_x[c] = (float) PSA_qraw[s][h][c];

        float rms = 0;
        for (int c = 0; c < head_dim; c++) {
            float x = vec_x[c];
            rms += (float) x * x;
        }
        rms = 1.0f / sqrtf(_1_dim * rms + norm_eps);

        // cal and store out
        for (int c = 0; c < head_dim; c++) 
            out[s][h][c] = (__bf16) (rms * vec_x[c]);
    }
}

void apply_rotary_emb(
    __bf16 in[seq_len][n_heads][head_dim],
    bool inverse
) {
    int rope_index = head_dim - rope_head_dim;
    float* WTB_freq_cis = &WTB[q_lora_rank * n_heads * head_dim + (q_lora_rank * n_heads * head_dim >> 14)];
    __bf16* RAB_q = &RAB[seq_len * q_lora_rank];

    // local buffer
    float freq_cis[seq_len][rope_head_dim];

    // load freq_cis
    for (int s = 0; s < seq_len; s++) 
    for (int r = 0; r < rope_head_dim; r++) 
        freq_cis[s][r] = WTB_freq_cis[s * rope_head_dim + r];

    for (int s = 0; s < seq_len; s++) 
    for (int h = 0; h < n_heads; h++) {
        // store nonrope part
        for (int c = 0; c < rope_index; c++) 
            RAB_q[(s * n_heads + h) * head_dim + c] = in[s][h][c];

        // rope
        for (int c = rope_index, f = 0; c < head_dim; c++, c++, f++, f++) {

            float cos_d = freq_cis[s][f];
            float sin_d = inverse ? -freq_cis[s][f + 1] : freq_cis[s][f + 1];
                                    
            float a = in[s][h][c], b = in[s][h][c + 1];

            // store to RAB
            RAB_q[(s * n_heads + h) * head_dim + c] = (__bf16) (a * cos_d - b * sin_d);
            RAB_q[(s * n_heads + h) * head_dim + c + 1] = (__bf16) (a * sin_d + b * cos_d);
        }
    }
}


void query_up(
    fp8 X[seq_len][q_lora_rank], 
    fp8 scale[seq_len][q_lora_rank >> 7]
) {
    __bf16 wire[seq_len][n_heads][head_dim];

    quant_gemm(X, scale);

    rms_norm(wire);
    
    apply_rotary_emb(wire, false);
}


void sparse_attention(
    __bf16 out[seq_len][n_heads][head_dim]
) {
    float scale = sqrtf(1.0f / head_dim);

    __bf16* RAB_q = &RAB[seq_len * q_lora_rank];
    float* WTB_attn_sink = &WTB[q_lora_rank * n_heads * head_dim + (q_lora_rank * n_heads * head_dim >> 14) + seq_len * rope_head_dim];

    __bf16 q_shared[n_heads][head_dim];
    __bf16 kv_shared[block][head_dim];
    float attn_sink[n_heads];
    __bf16 acc_s_cast[n_heads][block];

    int idxs[block];
    float acc_s[n_heads][block];
    float acc_o[n_heads][head_dim];
    float score_max[n_heads];
    float score_max_prev[n_heads];
    float score_scale[n_heads];
    float score_sum[n_heads];
    float sum_exp[n_heads];

    for (int i = 0; i < n_heads; i++) 
        attn_sink[i] = WTB_attn_sink[i];

    for (int s = 0; s < seq_len; s++) {

        // clear buffer, load q
        for (int h = 0; h < n_heads; h++) {
            sum_exp[h] = 0;
            score_max[h] = -1.0f / 0.0f;
            for (int c = 0; c < head_dim; c++) {
                acc_o[h][c] = 0;
                q_shared[h][c] = RAB_q[(s * n_heads + h) * head_dim + c];
            }
        }


        for (int t = 0; t < num_blocks; t++) {

            // load topk_idxs
            for (int i = 0; i < block; i++) 
                idxs[i] = t * block + i < K ? KVB_topk[s * K + t * block + i] : -1;

            // load kv from topk_idxs
            for (int i = 0; i < block; i++) 
            for (int c = 0; c < head_dim; c++) 
                kv_shared[i][c] = idxs[i] != -1 ? KVB_kv[idxs[i] * head_dim + c] : 0;
            
            for (int h = 0; h < n_heads; h++) 
            for (int i = 0; i < block; i++) 
                acc_s[h][i] = idxs[i] != -1 ? 0 : -1.0f / 0.0f;

            gemm<__bf16, float, n_heads, block, head_dim>(q_shared, (__bf16*) kv_shared, true, true, acc_s);

            for (int h = 0; h < n_heads; h++) 
            for (int i = 0; i < block; i++) 
                acc_s[h][i] *= scale;

            for (int h = 0; h < n_heads; h++) 
                score_max_prev[h] = score_max[h];

            for (int h = 0; h < n_heads; h++) {
                score_max[h] = -1.0f / 0.0f;
                for (int i = 0; i < block; i++) {
                    float cur = acc_s[h][i];
                    if (cur > score_max[h]) 
                        score_max[h] = cur;
                }
            }

            for (int h = 0; h < n_heads; h++) 
                score_scale[h] = expf(score_max_prev[h] - score_max[h]);

            for (int h = 0; h < n_heads; h++) 
            for (int i = 0; i < block; i++) 
                acc_s[h][i] = expf(acc_s[h][i] - score_max[h]);

            for (int h = 0; h < n_heads; h++) {
                score_sum[h] = 0;
                for (int i = 0; i < block; i++) 
                    score_sum[h] += acc_s[h][i];
            }

            for (int h = 0; h < n_heads; h++) 
                sum_exp[h] = sum_exp[h] * score_scale[h] + score_sum[h];

            for (int h = 0; h < n_heads; h++) 
            for (int i = 0; i < block; i++) 
                acc_s_cast[h][i] = (__bf16) acc_s[h][i];

            for (int h = 0; h < n_heads; h++) 
            for (int c = 0; c < head_dim; c++) 
                acc_o[h][c] *= score_scale[h];

            gemm<__bf16, float, n_heads, head_dim, block>(acc_s_cast, (__bf16*) kv_shared, false, true, acc_o);
        }

        for (int h = 0; h < n_heads; h++) 
            sum_exp[h] += expf(attn_sink[h] - score_max[h]);

        // cal and store out
        for (int h = 0; h < n_heads; h++) 
        for (int c = 0; c < head_dim; c++) {
            acc_o[h][c] /= sum_exp[h];
            out[s][h][c] = (__bf16) acc_o[h][c];
        }
    }
}


void QCAU(
    fp8 X[seq_len][q_lora_rank], 
    fp8 scale[seq_len][q_lora_rank >> 7]
) {
    __bf16 wire[seq_len][n_heads][head_dim];

    query_up(X, scale);
    sparse_attention(wire);
    apply_rotary_emb(wire, true);
}
