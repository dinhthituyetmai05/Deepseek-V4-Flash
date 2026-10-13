#include "include/datapath_mHCE.h"

// Khai báo buffer và thanh ghi nội
internal_registers_format internal_register;
internal_buffers_format internal_buffer;

void hyperconnection(
    bfloat16_e8m7 *i_x[N_HC][DIM],
    bfloat16_e8m7 *o_nx[DIM],
    int token_id
) {
    for (int token_id = 0; token_id < TOKEN; token_id++) {
        
        internal_register.sum_x_i_squared.fp32 = 0;
        RMSnorm(
            &internal_register,
            &i_x
        );

        for (int w_pre_id = 0; w_pre_id < N_HC*DIM; w_pre_id += WEIGHT_TILE_ROW) {
            load_w_pre(
                &mSGDMA,
                &WTB,
                w_pre_id
            );
            
            Sinh_A(
                &internal_register,
                axi4_lite
            );
        }

        for (int w_res_id = 0; w_res_id<N_HC*DIM; w_res_id += WEIGHT_TILE_ROW) {
            load_w_res(
                &mSGDMA,
                &WTB,
                w_res_id
            );

            Sinh_B(
                &internal_register,
                axi4_lite
            );

            store_B_i(
                &internal_buffer,
                &internal_register,
                token_id
            );
        }

        for (int w_post_id = 0; w_post_id<N_HC*DIM; w_post_id += WEIGHT_TILE_ROW) {
            load_w_post(
                &mSGDMA,
                &WTB,
                w_post_id
            );

            Sinh_C(
                &internal_register,
                axi4_lite
            );

            store_C_i(
                &internal_buffer,
                &internal_register,
                token_id
            );
        }

        compute_o_nx(
            &internal_register,
            &i_x,
            &o_nx
        );
    }
}

void residual_mixing(
    bfloat16_e8m7 *i_x[N_HC][DIM],
    bfloat16_e8m7 *i_f[DIM],
    bfloat16_e8m7 *o_x[N_HC][DIM],
    int token_id
) {
    load_B(
        &internal_buffer,
        &internal_register,
        token_id
    );

    compute_i_x_B(
        &internal_register,
        &i_x
    );

    load_C(
        &internal_buffer,
        &internal_register,
        token_id
    );

    compute_i_f_C(
        &internal_register,
        &i_f[DIM]
    );

    compute_o_x(
        &internal_register,
        &o_x
    );

    store_o_x(
        &DDR4,
        &o_x,
        token_id
    );
}


// Layer đầu tiên tensor load từ DDR4
void FirstLayer___Preblock_mHCE() {
    for (int token_id = 0; token_id < TOKEN; token_id++) {
        load_x_i(
            &DDR4,
            &RAB.i_x,
            token_id
        );

        hyperconnection(
            &RAB.i_x,
            &RAB.x_hat_attn[token_id],
            token_id
        );
    }
}

// Layer đầu tiên load từ DDR4
void FirstLayer___Residual_mixing_attn___Postblock_mHCE() {
    for (int token_id = 0; token_id < TOKEN; token_id++) {
        residual_mixing(
            &RAB.i_x,
            &RAB.x_out_attn[token_id],
            &RAB.o_attn[token_id],
            token_id
        );

        hyperconnection(
            &RAB.o_attn[token_id],
            &RAB.x_hat_moe[token_id],
            token_id
        );
    }
} 

// Residual mixing sau MOE của layer trước gộp với Preblock mHCE của layer sau
void PreviousLayer___Residual_mixing_attn___NextLayer___Preblock_mHCE() {
    for (int token_id = 0; token_id < TOKEN; token_id++) {
        residual_mixing(
            &RAB.o_attn[token_id],
            &RAB.x_out_moe[token_id],
            &RAB.o_moe[token_id],
            token_id
        );

        hyperconnection(
            &RAB.o_moe[token_id],
            &RAB.x_hat_attn[token_id],
            token_id
        );
    }
}

// Residual mixing sau attn gộp với Postblock mHCE của cùng 1 layer
void CurrentLayer___Residual_mixing_moe___Postblock_mHCE() {
    for (int token_id = 0; token_id < TOKEN; token_id++) {
        residual_mixing(
            &RAB.o_moe[token_id],
            &RAB.x_out_attn[token_id],
            &RAB.o_attn[token_id],
            token_id
        );

        hyperconnection(
            &RAB.o_attn[token_id],
            &RAB.x_hat_moe[token_id],
            token_id
        );
    }
}

// Layer cuối tensor store tensor từ RAB lên DDR4
void LastLayer___Residual_mixing_moe() {
    for (int token_id = 0; token_id < TOKEN; token_id++) {
        residual_mixing(
            &RAB.o_moe[token_id],
            &RAB.x_out_attn[token_id],
            &RAB.i_x,
            token_id
        );
        /*
        store_o_x(
            &DDR4,
            &RAB.i_x,
            token_id
        );
        */
    }
}