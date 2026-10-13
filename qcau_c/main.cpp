#include "include/Core.h"
#include "src/qcau.cpp"


int main() {
    int seq_len = 16, q_lora_rank = 1024, n_heads = 64, head_dim = 512, rope_head_dim = 64;
    int n = 20, k = 20, norm_eps = 1e-6;

    Tensor<__bf16> c_q({seq_len, 1, q_lora_rank});
    Tensor<float> c_q_scale({seq_len, 1, q_lora_rank >> 7});
    Tensor<__bf16> w_uq({1, n_heads * head_dim, q_lora_rank});
    Tensor<float> w_uq_scale({1, n_heads * head_dim >> 7, q_lora_rank >> 7});
    Tensor<__bf16> kv({1, n, head_dim});
    Tensor<int> topk_idxs({seq_len, 1, k});
    Tensor<float> attn_sink({1, n_heads, 1});
    Tensor<float> freq_cis({seq_len, 1, rope_head_dim});
    Tensor<__bf16> output({seq_len, n_heads, head_dim});

    //Tensor<__bf16> q({seq_len, n_heads, head_dim});

    read_tensor("data/hardware_seq16_layer2/linear_x_after_act_quant.fp8_e4m3.txt", c_q);
    read_tensor("data/hardware_seq16_layer2/linear_s_after_act_quant.e8m0.txt", c_q_scale);
    read_tensor("data/hardware_seq16_layer2/w_uq_weight.txt", w_uq);
    read_tensor("data/hardware_seq16_layer2/w_uq_scale.txt", w_uq_scale);
    read_tensor("data/hardware_seq16_layer2/kv_after_rope_quant_compress.bf16.txt", kv);
    read_tensor("data/hardware_seq16_layer2/topk_idxs.int.txt", topk_idxs);
    read_tensor("data/hardware_seq16_layer2/attn_sink.fp32.txt", attn_sink);
    read_tensor("data/hardware_seq16_layer2/freqs_cis_comp.fp32.txt", freq_cis);

    //read_tensor("data/hardware_seq16_layer2/q_after_query_up_rope.bf16.txt", q);

    //query_up(c_q, w_uq, c_q_scale, w_uq_scale, freq_cis, rope_head_dim, norm_eps, q);
    //sparse_attention(q, kv, attn_sink, topk_idxs, output);
    QCAU(c_q, w_uq, c_q_scale, w_uq_scale, kv, topk_idxs, attn_sink, freq_cis, head_dim, rope_head_dim, norm_eps, output);

    write_tensor("data/hardware_seq16_layer2/test_output.txt", output, 15);

    return 0;
}
