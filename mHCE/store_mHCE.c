#include "include/store_mHCE.h"

void store_B_i(
    internal_buffers_format *internal_buffer,
    internal_registers_format *internal_register,
    int token_id
) {
    memcpy(
        &internal_buffer->storage_B[token_id],  // (256,4,4)
        &internal_register->B_i,                //(4,4)
        sizeof(internal_buffer->storage_B[token_id])
    );
}

void store_C_i(
    internal_buffers_format *internal_buffer,
    internal_registers_format *internal_register,
    int token_id
) {
    memcpy(
        &internal_buffer->storage_C[token_id],  // (256,4,1)
        &internal_register->C_i,                //(4,1)
        sizeof(internal_buffer->storage_C[token_id])
    );
}

void store_o_x(
    DDR4_format *DDR4,
    bfloat16_e8m7 o_x[N_HC][DIM],
    int token_id
) {
    memcpy(
        &DDR4->o_x[token_id],
        &o_x,
        sizeof(DDR4->o_x[token_id])
    );
}