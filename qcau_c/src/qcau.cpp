#include "../include/Core.h"


template <typename T1, typename T2>
void gemm(
    Tensor<T1> &X1, Tensor<T1> &X2, 
    bool transpose, bool accum, 
    Tensor<T2> &Y
) {
    if (transpose && (X1.shape.C != X2.shape.C || Y.shape.C != X2.shape.H) 
        || !transpose && (X1.shape.C != X2.shape.H || Y.shape.C != X2.shape.C) 
        || (Y.shape.H != X1.shape.H || X1.shape.S != X2.shape.S || X1.shape.S != Y.shape.S)
    ) 
        assert(0 && "Error: Inappropriate size for gemm !!");

    int H = Y.shape.H, C = Y.shape.C, K = X1.shape.C, S = Y.shape.S;

    for (int s = 0; s < S; s++) 
    for (int i = 0; i < H; i++) 
    for (int j = 0; j < C; j++) {
        if (!accum) 
            Y.at(s, i, j) = 0;
        for (int k = 0; k < K; k++) {
            float B = transpose ? X2.at(s, j, k) : X2.at(s, k, j);
            Y.at(s, i, j) += (T2) (X1.at(s, i, k) * B);
        }
    }
}

template <typename T1, typename T2>
void quant_gemm(
    Tensor<T1> &X1, Tensor<T1> &X2, 
    Tensor<float> &scale_1, Tensor<float> &scale_2, 
    bool transpose, bool accum, 
    Tensor<T2> &Y
) {
    if (X2.shape.H >> 7 != scale_2.shape.H || X2.shape.C >> 7 != scale_2.shape.C 
        || X1.shape.H != scale_1.shape.H || X1.shape.C >> 7 != scale_1.shape.C
    )
        assert(0 && "Error: Inappropriate size for quant gemm !!");

    int H = scale_1.shape.H, C_block = transpose ? scale_2.shape.H : scale_2.shape.C;
    int K_block = scale_1.shape.C;
    int S = Y.shape.S;

    Tensor<float> x1({1, H, 128});
    Tensor<float> x2({1, 128, 128});
    Tensor<float> y({1, H, 128});
    Tensor<float> acc({1, H, 128});

    for (int s = 0; s < S; s++) 
    for (int c = 0; c < C_block; c++) {
        for (int k = 0; k < K_block; k++) {

            // load x1
            for (int i = 0; i < H; i++) 
            for (int j = 0; j < 128; j++) 
                x1.at(0, i, j) = X1.at(s, i, (k << 7) + j);
            
            // load x2
            for (int i = 0; i < 128; i++) 
            for (int j = 0; j < 128; j++) 
                x2.at(0, i, j) = transpose ? X2.at(s, (c << 7) + j, (k << 7) + i) 
                                            : X2.at(s, (k << 7) + i, (c << 7) + j);

            // cal y
            gemm(x1, x2, false, false, y);

            // acc += y
            for (int i = 0; i < H; i++) 
            for (int j = 0; j < 128; j++) {
                float s2 = transpose ? scale_2.at(0, c, k) : scale_2.at(0, k, c);
                float acc_num = y.at(0, i, j) * (scale_1.at(0, i, k) * s2);
                if (k) 
                    acc.at(0, i, j) += acc_num;
                else 
                    acc.at(0, i, j) = acc_num;
            }
        }
        
        // store acc
        for (int i = 0; i < H; i++) 
        for (int j = 0; j < 128; j++) {
            if (accum) 
                Y.at(s, i, (c << 7) + j) = (T2) (Y.at(s, i, (c << 7) + j) + acc.at(0, i, j));
            else 
                Y.at(s, i, (c << 7) + j) = (T2) acc.at(0, i, j);
        }
    }
}

template <typename T>
void rms_norm(
    Tensor<T> &X, Tensor<T>* gamma, float eps, 
    Tensor<T> &Y
) {
    if (gamma && (X.shape.C != gamma->shape.C) || !(X.shape == Y.shape)) 
        assert(0 && "Error: Inappropriate size for norm !!");

    int H = Y.shape.H, C = Y.shape.C, S = Y.shape.S;
    Tensor<float> rms({X.shape.S, X.shape.H, 1});
    float _1_dim = 1.0f / C;

    for (int s = 0; s < S; s++) 
    for (int h = 0; h < H; h++) {
        rms.at(s, h, 0) = 0;
        for (int c = 0; c < C; c++) {
            T x = X.at(s, h, c);
            rms.at(s, h, 0) += (float) x * x;
        }
        rms.at(s, h, 0) = 1.0f / sqrtf(_1_dim * rms.at(s, h, 0) + eps);

        for (int c = 0; c < C; c++) {
            Y.at(s, h, c) = (T) (rms.at(s, h, 0) * X.at(s, h, c));

            if (gamma) 
                Y.at(s, h, c) = (T) (Y.at(s, h, c) * (float) gamma->at(0, 0, c));
        }
    }
}

template <typename T>
void apply_rotary_emb(
    Tensor<T> &X, Tensor<float> &freq_cis, 
    int rope_head_dim, bool inverse, 
    Tensor<T> &Y
) {
    if ((rope_head_dim & 1) || X.shape.C < rope_head_dim || freq_cis.shape.S != X.shape.S
        || freq_cis.shape.C != rope_head_dim || !(X.shape == Y.shape)
    ) 
        assert(0 && "Error: Inappropriate size for rope !!");

    int H = Y.shape.H, C = Y.shape.C, S = Y.shape.S;
    int rope_index = C - rope_head_dim;

    for (int s = 0; s < S; s++) 
    for (int h = 0; h < H; h++) 
    for (int c = rope_index, f = 0; c < C; c++, c++, f++, f++) {
        float cos_d = freq_cis.at(s, 0, f);
        float sin_d = inverse ? -freq_cis.at(s, 0, f + 1) : freq_cis.at(s, 0, f + 1);
        float a = X.at(s, h, c), b = X.at(s, h, c + 1);

        Y.at(s, h, c) = (T) (a * cos_d - b * sin_d);
        Y.at(s, h, c + 1) = (T) (a * sin_d + b * cos_d);
    }
}

template <typename T>
void query_up(
    Tensor<T> &c_q, Tensor<T> &w_uq, 
    Tensor<float> &c_q_scale, Tensor<float> &w_uq_scale, 
    Tensor<float> &freq_cis, 
    int rope_head_dim, float norm_eps, 
    Tensor<T> &output_q
) {
    int head_dim = output_q.shape.C;

    output_q.reshape(1, output_q.shape.S, head_dim * output_q.shape.H);
    c_q.reshape(1, c_q.shape.S, c_q.shape.C);
    c_q_scale.reshape(1, c_q_scale.shape.S, c_q_scale.shape.C);
    quant_gemm(c_q, w_uq, c_q_scale, w_uq_scale, true, false, output_q);

    output_q.reshape(output_q.shape.H, output_q.shape.C / head_dim, head_dim);
    rms_norm(output_q, (Tensor<T>*) NULL, norm_eps, output_q);
    
    apply_rotary_emb(output_q, freq_cis, rope_head_dim, false, output_q);
}

template <typename T>
void sparse_attention(
    Tensor<T> &q, Tensor<T> &kv, 
    Tensor<float> &attn_sink, Tensor<int> &topk_idx, 
    Tensor<T> &out
) {
    if (q.shape.S != topk_idx.shape.S || q.shape.C != kv.shape.C || q.shape.H != attn_sink.shape.H
        || out.shape.S != q.shape.S || out.shape.H != q.shape.H || out.shape.C != q.shape.C
    ) 
        assert(0 && "Error: Inappropriate size for sparse attn !!");

    int H = out.shape.H, C = out.shape.C, S = out.shape.S, topk = topk_idx.shape.C;
    float scale = sqrtf(1.0f / q.shape.C);

    int block = 64;
    int num_blocks = ceilf((float) topk / block);

    Tensor<__bf16> q_shared({1, H, C});
    Tensor<__bf16> kv_shared({1, block, C});
    Tensor<__bf16> acc_s_cast({1, H, block});

    int* idxs = (int*)malloc(block * sizeof(int));
    Tensor<float> acc_s({1, H, block});
    Tensor<float> acc_o({1, H, C});
    float* score_max = (float*)malloc(H * sizeof(float));
    float* score_max_prev = (float*)malloc(H * sizeof(float));
    float* score_scale = (float*)malloc(H * sizeof(float));
    float* score_sum = (float*)malloc(H * sizeof(float));
    float* sum_exp = (float*)malloc(H * sizeof(float));

    for (int s = 0; s < S; s++) {

        for (int h = 0; h < H; h++) {
            sum_exp[h] = 0;
            score_max[h] = -1.0f / 0.0f;
            for (int c = 0; c < C; c++) {
                acc_o.at(0, h, c) = 0;
                q_shared.at(0, h, c) = q.at(s, h, c);
            }
        }


        for (int t = 0; t < num_blocks; t++) {
            for (int i = 0; i < block; i++) 
                idxs[i] = t * block + i < topk ? topk_idx.at(s, 0, t * block + i) : -1;

            for (int i = 0; i < block; i++) 
            for (int c = 0; c < C; c++) 
                kv_shared.at(0, i, c) = idxs[i] != -1 ? kv.at(0, idxs[i], c) : 0;
            
            for (int h = 0; h < H; h++) 
            for (int i = 0; i < block; i++) 
                acc_s.at(0, h, i) = idxs[i] != -1 ? 0 : -1.0f / 0.0f;

            gemm(q_shared, kv_shared, true, true, acc_s);

            for (int h = 0; h < H; h++) 
            for (int i = 0; i < block; i++) 
                acc_s.at(0, h, i) *= scale;

            for (int h = 0; h < H; h++) 
                score_max_prev[h] = score_max[h];

            for (int h = 0; h < H; h++) {
                score_max[h] = -1.0f / 0.0f;
                for (int i = 0; i < block; i++) {
                    float cur = acc_s.at(0, h, i);
                    if (cur > score_max[h]) 
                        score_max[h] = cur;
                }
            }

            for (int h = 0; h < H; h++) 
                score_scale[h] = expf(score_max_prev[h] - score_max[h]);

            for (int h = 0; h < H; h++) 
            for (int i = 0; i < block; i++) 
                acc_s.at(0, h, i) = expf(acc_s.at(0, h, i) - score_max[h]);

            for (int h = 0; h < H; h++) {
                score_sum[h] = 0;
                for (int i = 0; i < block; i++) 
                    score_sum[h] += acc_s.at(0, h, i);
            }

            for (int h = 0; h < H; h++) 
                sum_exp[h] = sum_exp[h] * score_scale[h] + score_sum[h];

            for (int h = 0; h < H; h++) 
            for (int i = 0; i < block; i++) 
                acc_s_cast.at(0, h, i) = (__bf16) acc_s.at(0, h, i);

            for (int h = 0; h < H; h++) 
            for (int c = 0; c < C; c++) 
                acc_o.at(0, h, c) *= score_scale[h];

            gemm(acc_s_cast, kv_shared, false, true, acc_o);
        }

        for (int h = 0; h < H; h++) 
            sum_exp[h] += expf(attn_sink.at(0, h, 0) - score_max[h]);

        for (int h = 0; h < H; h++) 
        for (int c = 0; c < C; c++) {
            acc_o.at(0, h, c) /= sum_exp[h];
            out.at(s, h, c) = (T) acc_o.at(0, h, c);
        }
    }

    delete idxs;
    delete score_max;
    delete score_max_prev;
    delete score_scale;
    delete score_sum;
    delete sum_exp;
}

template <typename T>
void QCAU(
    Tensor<T> &c_q, Tensor<T> &w_uq, 
    Tensor<float> &c_q_scale, Tensor<float> &w_uq_scale, 
    Tensor<T> &kv, 
    Tensor<int> &topk_idxs, Tensor<float> &attn_sink, 
    Tensor<float> &freq_cis, 
    int head_dim, int rope_head_dim, float norm_eps, 
    Tensor<T> &out
) {
    Tensor<T> q({c_q.shape.S, w_uq.shape.H / head_dim, head_dim});
    query_up(c_q, w_uq, c_q_scale, w_uq_scale, freq_cis, rope_head_dim, norm_eps, q);
    sparse_attention(q, kv, attn_sink, topk_idxs, out);
    apply_rotary_emb(out, freq_cis, rope_head_dim, true, out);
}
