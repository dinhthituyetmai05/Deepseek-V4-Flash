#include "include/kernel_mHCE.h"

/*
Convention:
    float           ->  float
    float32_e8m23   ->  fp32
    bfloat16_e8m7   ->  bf16
*/

// Function để thực hiện chuyển đổi từ bit-level sang float của C
float conv_float_from_bits(uint32_t bits) {
    float fp;
    memcpy(
        &fp,
        &bits,
        sizeof(float)
    );
    return fp;
}

// Function để thực hiện chuyển đổi từ float của C sang bit-level
uint32_t conv_bits_from_float(float fp) {
    uint32_t bits;
    memcpy(
        &bits,
        &fp,
        sizeof(float)
    );
    return bits;
}

float32_e8m23 cast_fp32_from_bf16(
    bfloat16_e8m7 *input
) {
    float32_e8m23 output;

    output.bits.mantissa = input->bits.mantissa << 16;
    output.bits.exponent = input->bits.exponent;
    output.bits.sign = input->bits.sign;

    return output;
}

bfloat16_e8m7 cast_bf16_from_fp32(
    float32_e8m23 *input
) {
    bfloat16_e8m7 output;

    output.bits.mantissa = input->bits.mantissa >> 16;
    output.bits.exponent = input->bits.exponent;
    output.bits.sign = input->bits.sign;

    return output;
}

void accumulate(
    bfloat16_e8m7 *i_x,             //Từ RAB
    float32_e8m23 *x_i_flattened,
    float32_e8m23 *sum_x_i_squared
) { 

    // 1. Cast BF16->FP32
    // 2. Thực hiện phép toán sum += x_i_flatened^2

    *x_i_flattened = cast_fp32_from_bf16(i_x);

    //Thực hiện phép toán,
    float x_i_flat = conv_float_from_bits(x_i_flattened->fp32);
    float sum = conv_float_from_bits(sum_x_i_squared->fp32);

    sum += x_i_flat * x_i_flat;
    sum_x_i_squared->fp32 = conv_bits_from_float(sum);
}

void customized_sqrt(
    float32_e8m23 *sum_x_i_squared
) {

    // 1. Chuyển bit-level sang float của C
    // 2. thực hiện căn bậc 2
    // 3. Chuyển float của C về bit level

    // multiply by 1/16384 = 2^-14, subtract 14 from the exponent
    sum_x_i_squared->bits.exponent -= 14; //Có thể bị bug nếu exponent bé hơn 14 - Fix sau
    
    float result = conv_float_from_bits(sum_x_i_squared->fp32) + 1e-6f;
    result = sqrtf(result);
    sum_x_i_squared->fp32 = conv_bits_from_float(result);
}

void normalization(
    float32_e8m23 *x_i_flattened, //Từng phần tử của x_i_flattened
    float32_e8m23 *sum_x_i_squared,
    float32_e8m23 *gamma,
    float32_e8m23 *x_nga_i
) {

    // Thực hiện phép toán : scalar * x_i / sum

    float x_i = conv_float_from_bits(x_i_flattened->fp32);
    float scalar = conv_float_from_bits(gamma->fp32);
    float sum = conv_float_from_bits(sum_x_i_squared->fp32);

    x_i = scalar * (x_i / sum);

    x_nga_i->fp32 = conv_bits_from_float(x_i);

}

void matrix_multiplication_fp32(
    float32_e8m23 *x_i,
    float32_e8m23 *w_i,
    float32_e8m23 *result
) {
    float x_fp = conv_float_from_bits(x_i->fp32);
    float w_fp = conv_float_from_bits(w_i->fp32);
    float result_fp = conv_float_from_bits(result->fp32);

    result_fp += x_fp * w_fp;

    result->fp32 = conv_bits_from_float(result_fp);
}

void elementwise_multiplication_operation(
    float32_e8m23 *he_so,
    float32_e8m23 *scalar_nhan
) {
    float he_so_fp = conv_float_from_bits(he_so->fp32);
    float scalar_nhan_fp = conv_float_from_bits(scalar_nhan->fp32);

    he_so_fp *= scalar_nhan_fp;

    he_so->fp32 = conv_bits_from_float(he_so_fp);
}

void elementwise_addition_operation(
    float32_e8m23 *he_so,
    float32_e8m23 *scalar_cong
) {
    float he_so_fp = conv_float_from_bits(he_so->fp32);
    float scalar_cong_fp = conv_float_from_bits(scalar_cong->fp32);

    he_so_fp += scalar_cong_fp;

    he_so->fp32 = conv_bits_from_float(he_so_fp);
}

void ham_sigmoid(
    float32_e8m23 *he_so,
    float scale
) {
    float he_so_float = conv_float_from_bits(he_so->fp32);
    he_so_float = 1.0f / (1.0f + expf(-he_so_float)); // sigmoid
    he_so_float = scale * he_so_float;
    he_so->fp32 = conv_bits_from_float(he_so_float);
}

void SK_algorithm(
    internal_registers_format *internal_register
) {
    float B_fp[N_HC][N_HC] = {0};

    // FP32 bit-level -> C float
    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < N_HC; j++) {
            B_fp[i][j] = conv_float_from_bits(internal_register->B_i[i][j].fp32);
        }
    }

    for (int iter = 0; iter < SK_ROUND; iter++) {
        // Column normalization
        float sum_cols[N_HC] = {0};

        for (int i = 0; i < N_HC; i++) {
            for (int j = 0; j < N_HC; j++) {
                sum_cols[i] += B_fp[j][i];
            }
        }

        for (int i = 0; i < N_HC; i++) {
            for (int j = 0; j < N_HC; j++) {
                B_fp[j][i] = B_fp[j][i] / sum_cols[i];
            }
        }

        // Row normalization
        float sum_rows[N_HC] = {0};

        for (int i = 0; i < N_HC; i++) {
            for (int j = 0; j < N_HC; j++) {
                sum_rows[i] += B_fp[i][j];
            }
        }

        for (int i = 0; i < N_HC; i++) {
            for (int j = 0; j < N_HC; j++) {
                B_fp[i][j] = B_fp[i][j] / sum_rows[i];
            }
        }
    }

    // C float -> FP32 bit-level
    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < N_HC; j++) {
            internal_register->B_i[i][j].fp32 = conv_bits_from_float(B_fp[i][j]);
        }
    }
}

void RMSnorm(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *i_x[N_HC][DIM]
) {
    //Tổng bình phương 
    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < DIM; j++) {
            accumulate(
                &i_x[i][j],
                &internal_register->x_i_flattened[i * DIM + j],
                &internal_register->sum_x_i_squared
            );
        }
    }

    //chia 1/16384 và Căn bậc 2
    customized_sqrt(
        &internal_register->sum_x_i_squared
    );

    //Chuẩn hoá từng phần tử
    for (int i = 0; i < N_HC*DIM; i++) {
        normalization(
            &internal_register->x_i_flattened[i],
            &internal_register->sum_x_i_squared,
            &axi4_lite.registers.gamma,
            &internal_register->x_nga_i[i]
        );
    }
}

void sinh_A(
    internal_registers_format *internal_register
) {
    //Matrix Multiplication
    for (int i = 0; i < WEIGHT_TILE_ROW; i++) {
        for (int j = 0; j < WEIGHT_PRE_TILE_COL; j++) {
            matrix_multiplication_fp32(
                &internal_register->x_nga_i[i],
                &WTB.w_pre[i][j],
                &internal_register->A_i[0][j]
            );
        }
    }
    
    //Elementwise + Sigmoid
    for (int i = 0; i < N_HC; i++) {
        elementwise_multiplication_operation(
            &internal_register->A_i[0][i],
            &axi4_lite.registers.a_pre[0][i]
        );
        
        elementwise_addition_operation(
            &internal_register->A_i[0][i],
            &axi4_lite.registers.s_pre[0][i]
        );

        //Sigmoid
        ham_sigmoid(
            &internal_register->A_i[0][i],
            1.0f
        );
    }
}

void sinh_B(
    internal_registers_format *internal_register
) {
    //Matrix Multiplication
    for (int i = 0; i < WEIGHT_TILE_ROW; i++) {
        for (int j = 0; j < WEIGHT_RES_TILE_COL; j++) {
            matrix_multiplication(
                &internal_register->x_nga_i[i],
                &WTB.w_res[i][j],
                &internal_register->B_nga[0][j]
            );
        }
    }
    
    //reshape (1,16) -> (4,4)
    memcpy(
        internal_register->B_i,
        internal_register->B_nga,
        sizeof(internal_register->B_i)
    );

    //Elementwise operation
    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < N_HC; j++) {
            elementwise_multiplication_operation(
                &internal_register->B_i[i][j],
                &axi4_lite.registers.a_res[i][j]
            );
            elementwise_addition_operation(
                &internal_register->B_i[i][j],
                &axi4_lite.registers.s_res[i][j]
            );
        }
    }

    SK_algorithm(
        internal_register
    );
}

void sinh_C(
    internal_registers_format *internal_register
) {
    //Matrix Multiplication
    for (int i = 0; i < WEIGHT_TILE_ROW; i++) {
        for (int j = 0; j < WEIGHT_POST_TILE_COL; j++) {
            matrix_multiplication(
                &internal_register->x_nga_i[i],
                &WTB.w_post[i][j],
                &internal_register->C_nga[0][j]
            );
        }
    }
    
    //Transpose
    for (int i = 0; i < N_HC; i++) {
        internal_register->C_i[i][0].fp32 = internal_register->C_nga[0][i].fp32;
    }

    //Elementwise + 2*Sigmoid
    for (int i = 0; i < N_HC; i++) {
        elementwise_multiplication_operation(
            &internal_register->C_i[i][0],
            &axi4_lite.registers.a_post[i][0]
        );

        elementwise_operation(
            &internal_register->C_i[i][0],
            &axi4_lite.registers.s_post[i][0]
        );

        //Sigmoid
        ham_sigmoid(
            &internal_register->C_i[i][0],
            2.0f
        );
    }
}

void compute_o_nx(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *i_x[N_HC][DIM],
    bfloat16_e8m7 *o_nx[DIM]
) {

    // 1. Cast BF16 -> FP32
    // 2. Thực hiện matmul
    // 3. Cast kết quả ở FP32 -> BF16

    float32_e8m23 o_nx_fp32[DIM];

    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < DIM; j++) {
            float32_e8m23 x_i_fp32;
            x_i_fp32 = cast_fp32_from_bf16(i_x[i][j]->bf16);

            matrix_multiplication_fp32(
                &x_i_fp32,
                &internal_register->A_i[0][i],
                &o_nx_fp32[j]
            );
        }
    }

    for (int i = 0; i < DIM; i++) {
        *o_nx[i] = cast_bf16_from_fp32(o_nx_fp32[i].fp32);
    }
}

void compute_i_x_B(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *i_x[N_HC][DIM]
) {

    // 1. Cast BF16 -> FP32
    // 2. Thực hiện matmul

    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < N_HC; j++) {
            for(int k = 0; k < DIM; k++) {
                float32_e8m23 x_i_fp32;
                x_i_fp32 = cast_fp32_from_bf16(i_x[j][k]->bf16);

                matrix_multiplication_fp32(
                    &x_i_fp32,
                    &internal_register->B_i[i][j],
                    &internal_register->i_x_B[i][k]
                );
            }
        }
    }
}


void compute_i_f_C(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *i_f[DIM]
) {

    // 1. Cast BF16 -> FP32
    // 2. Thực hiện matmul

    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < DIM; j++) {
            float32_e8m23 x_i_fp32;
            x_i_fp32 = cast_fp32_from_bf16(i_f[j]->bf16);

            matrix_multiplication_fp32(
                &x_i_fp32,
                &internal_register->C_i[i][0],
                &internal_register->i_f_C[i][j]
            );
        }
    }
}

void compute_o_x(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *o_x[DIM]
) {

    //1. Elementwise addition
    //2. Cast FP32 -> BF16

    for (int i = 0; i < N_HC; i++) {
        for (int j = 0; j < DIM; j++) {
            elementwise_addition_operation(
                &internal_register->i_x_B[i][j],
                &internal_register->i_f_C[i][j]
            );

            o_x[i][j] = cast_bf16_from_fp32(&internal_register->i_x_B[i][j]);
        }
    }

}
