#ifndef LOAD_MHCE_H
#define LOAD_MHCE_H

#include <string.h>
#include "tensor_init.h"
#include "datapath_mHCE.h"

void load_x_i(
    DDR4_format *DDR4,
    bfloat16_e8m7 i_x[N_HC][DIM],
    int token_id
);

void load_w_pre(
    mSGDMA_format *mSGDMA,
    WTB_format *WTB,
    int w_pre_id
);
void load_w_res(
    mSGDMA_format *mSGDMA,
    WTB_format *WTB,
    int w_pre_id
);

void load_w_post(
    mSGDMA_format *mSGDMA,
    WTB_format *WTB,
    int w_post_id
);

void load_B_i(
    internal_buffers_format *internal_buffer,
    internal_registers_format *internal_register,
    int token_id
);

void load_C_i(
    internal_buffers_format *internal_buffer,
    internal_registers_format *internal_register,
    int token_id
);

#endif