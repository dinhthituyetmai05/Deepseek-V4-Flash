#include "include/load_mHCE.h"

void load_x_i(
    DDR4_format *DDR4,
    bfloat16_e8m7 i_x[N_HC][DIM],
    int token_id
) {
    memcpy(
        &i_x[N_HC][DIM],
        &DDR4->i_x[token_id],
        sizeof(i_x[N_HC][DIM])
    );
}

void load_w_pre(
    mSGDMA_format *mSGDMA,
    WTB_format *WTB,
    int w_pre_id
) {
    memcpy(
        &WTB->w_pre,
        &mSGDMA->w_pre[WEIGHT_TILE_ROW * w_pre_id],
        sizeof(WTB->w_pre)
    );
}

void load_w_res(
    mSGDMA_format *mSGDMA,
    WTB_format *WTB,
    int w_res_id
) {
    memcpy(
        &WTB->w_res,
        &mSGDMA->w_res[WEIGHT_TILE_ROW * w_res_id],
        sizeof(WTB->w_res)
    );
}

void load_w_post(
    mSGDMA_format *mSGDMA,
    WTB_format *WTB,
    int w_post_id
) {
    memcpy(
        &WTB->w_post,
        &mSGDMA->w_post[WEIGHT_TILE_ROW * w_post_id],
        sizeof(WTB->w_post)
    );
}


void load_B_i(
    internal_buffers_format *internal_buffer,
    internal_registers_format *internal_register,
    int token_id
) {
    memcpy(
        &internal_register->B_i,
        &internal_buffer->storage_B[token_id],
        sizeof(internal_register->B_i)
    );
}

void load_C_i(
    internal_buffers_format *internal_buffer,
    internal_registers_format *internal_register,
    int token_id
) {
    memcpy(
        &internal_register->C_i,
        &internal_buffer->storage_C[token_id],
        sizeof(internal_register->C_i)
    );
}