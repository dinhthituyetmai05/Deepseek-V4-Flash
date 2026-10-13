#include "include/Core.h"
#include "src/qcau_hw.cpp"


int main() {
    // int seq_len = 16, q_lora_rank = 1024, n_heads = 64, head_dim = 512, rope_head_dim = 64;
    // int n = 20, k = 20, norm_eps = 1e-6;

    float c_q[seq_len][q_lora_rank];
    float c_q_scale[seq_len][q_lora_rank >> 7];

    //Tensor<__bf16> q({seq_len, n_heads, head_dim});

    read_arr("data/hardware_seq16_layer2/linear_x_after_act_quant.fp8_e4m3.txt", (float*) c_q, seq_len * q_lora_rank);
    read_arr("data/hardware_seq16_layer2/linear_s_after_act_quant.e8m0.txt", (float*) c_q_scale, seq_len * q_lora_rank >> 7);
    read_arr("data/hardware_seq16_layer2/w_uq_weight.txt", WTB, q_lora_rank * n_heads * head_dim);
    read_arr("data/hardware_seq16_layer2/w_uq_scale.txt", &WTB[q_lora_rank * n_heads * head_dim], q_lora_rank * n_heads * head_dim >> 14);
    read_arr("data/hardware_seq16_layer2/kv_after_rope_quant_compress.bf16.txt", KVB_kv, N * head_dim);
    read_arr("data/hardware_seq16_layer2/topk_idxs.int.txt", KVB_topk, seq_len * K);
    read_arr("data/hardware_seq16_layer2/attn_sink.fp32.txt", &WTB[q_lora_rank * n_heads * head_dim + (q_lora_rank * n_heads * head_dim >> 14) + seq_len * rope_head_dim], n_heads);
    read_arr("data/hardware_seq16_layer2/freqs_cis_comp.fp32.txt", &WTB[q_lora_rank * n_heads * head_dim + (q_lora_rank * n_heads * head_dim >> 14)], seq_len * rope_head_dim);

    //read_tensor("data/hardware_seq16_layer2/q_after_query_up_rope.bf16.txt", q);

    //query_up(c_q, w_uq, c_q_scale, w_uq_scale, freq_cis, rope_head_dim, norm_eps, q);
    //sparse_attention(q, kv, attn_sink, topk_idxs, output);
    QCAU(c_q, c_q_scale);

    write_arr("data/hardware_seq16_layer2/test_output_0.txt", &RAB[seq_len * q_lora_rank], seq_len * n_heads * head_dim, 15);
    //write_arr("data/hardware_seq16_layer2/test_output_0.txt", (__bf16*) PSA_qraw, seq_len * n_heads * head_dim, 15);
    //write_arr("data/hardware_seq16_layer2/test_output_0.txt", (__bf16*) out_0, seq_len * n_heads * head_dim, 15);
    return 0;
}
