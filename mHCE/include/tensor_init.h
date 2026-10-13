#ifndef TENSOR_INIT_H
#define TENSOR_INIT_H

#include <stdint.h>
#include "config_mHCE.h"
// Định nghĩa FP32, BF16
// Định nghĩa từng bit do phép toán trên floating-point và brain fp thực hiện trên từng bit
typedef union {
    uint32_t fp32;
    struct {
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } bits;
} float32_e8m23;

typedef union {
    uint16_t bf16;
    struct {
        uint16_t mantissa : 7;
        uint16_t exponent : 8;
        uint16_t sign : 1;
    } bits;
} bfloat16_e8m7;

// Định nghĩa thanh ghi của axi4-lite, truyền từ control sequencer
typedef union {
    struct {
        float32_e8m23 gamma;

        float32_e8m23 a_pre[1][N_HC];
        float32_e8m23 a_res[N_HC][N_HC];
        float32_e8m23 a_post[N_HC][1];

        float32_e8m23 s_pre[1][N_HC];
        float32_e8m23 s_res[N_HC][N_HC];
        float32_e8m23 s_post[N_HC][1];

    } registers;
} axi4_lite_format;

// Định nghĩa DDR4, bên trong chứa các tensor đầu vào, 
// Kích thước có thể thay đổi do chứa tensor khác
typedef struct {
    bfloat16_e8m7 i_x[TOKEN][N_HC][DIM];
    bfloat16_e8m7 o_x[TOKEN][N_HC][DIM];

} DDR4_format;

// Định nghĩa mSGDMA, bên trong chưa các weight tensor đầu vào
// Kích thước có thể thay đổi do chứa tensor weight khác
typedef struct {
    float32_e8m23 w_pre[N_HC * DIM][N_HC];
    float32_e8m23 w_res[N_HC * DIM][N_HC*N_HC];
    float32_e8m23 w_post[N_HC * DIM][N_HC];

} mSGDMA_format;

// Định nghĩa Row Activation Buffer
typedef struct {
    bfloat16_e8m7 i_x[N_HC][DIM]; // Tensor đầu vào của Hyperconnection

    bfloat16_e8m7 x_hat_attn[TOKEN][DIM]; // Tensor output của Hyperconnection, Input của khối attn
    bfloat16_e8m7 x_out_attn[TOKEN][DIM]; // Tensor output của khối attn

    bfloat16_e8m7 c_q[TOKEN][1024];
    bfloat16_e8m7 Q[TOKEN][64][512];           // Không rõ
    bfloat16_e8m7 O_qcau[TOKEN][64][512];

    bfloat16_e8m7 o_attn[TOKEN][N_HC][DIM]; // Tensor output của residual mixing cho attn

    bfloat16_e8m7 x_hat_moe[TOKEN][DIM]; // Tensor output của Hyperconnection, Input của khối moe
    bfloat16_e8m7 x_out_moe[TOKEN][DIM]; // Tensor output của khối moe

    bfloat16_e8m7 o_moe[TOKEN][N_HC][DIM]; // Tensor output của residual mixing cho moe

} RAB_format; //RAB cho 1 layer, chưa tính chạy layer sau

// Định nghĩa Weight Tile Buffer
typedef struct {
    float32_e8m23 w_pre[WEIGHT_TILE_ROW][WEIGHT_PRE_TILE_COL];
    float32_e8m23 w_res[WEIGHT_TILE_ROW][WEIGHT_RES_TILE_COL];
    float32_e8m23 w_post[WEIGHT_TILE_ROW][WEIGHT_POST_TILE_COL];

} WTB_format;

// Định nghĩa thanh ghi nội
typedef struct {
    float32_e8m23 x_i_flattened[N_HC * DIM];
    float32_e8m23 sum_x_i_squared;
    float32_e8m23 x_nga_i[N_HC * DIM];

    float32_e8m23 B_nga[1][N_HC * N_HC];
    float32_e8m23 C_nga[1][N_HC];

    float32_e8m23 A_i[1][N_HC];
    float32_e8m23 B_i[N_HC][N_HC];
    float32_e8m23 C_i[N_HC][1];

    float32_e8m23 i_x_B[N_HC][DIM];
    float32_e8m23 i_f_C[N_HC][DIM];

    float32_e8m23 o_nx[DIM];
    
} internal_registers_format;

// Định nghĩa buffer nội
typedef struct {
    float32_e8m23 storage_B[TOKEN][N_HC][N_HC];
    float32_e8m23 storage_C[TOKEN][N_HC][1];

} internal_buffers_format;

// Khai báo global buffer
RAB_format RAB;
WTB_format WTB;
axi4_lite_format axi4_lite;

extern DDR4_format DDR4;
extern mSGDMA_format mSGDMA;

void tensor_init();

#endif