#ifndef STORE_MHCE_H
#define STORE_MHCE_H

#include <string.h>
#include "tensor_init.h"

void store_B_i(
    internal_buffers_format *internal_buffer,
    internal_registers_format *internal_register,
    int token_id
);

void store_C_i(
    internal_buffers_format *internal_buffer,
    internal_registers_format *internal_register,
    int token_id
);

void store_o_x(
    DDR4_format *DDR4,
    bfloat16_e8m7 o_x[N_HC][DIM],
    int token_id
);

#endif