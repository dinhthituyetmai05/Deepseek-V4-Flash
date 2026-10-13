#ifndef KERNEL_MHCE_H
#define KERNEL_MHCE_H

#include <string.h>
#include <math.h>
#include "tensor_init.h"

void RMSnorm(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *i_x[N_HC][DIM]
);

void sinh_A(
    internal_registers_format *internal_register
);

void sinh_B(
    internal_registers_format *internal_register
);

void sinh_C(
    internal_registers_format *internal_register
);

void compute_o_nx(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *i_x[N_HC][DIM],
    bfloat16_e8m7 *o_nx[DIM]
);

void compute_i_x_B(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *i_x[N_HC][DIM]
);


void compute_i_f_C(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *i_f[DIM]
);

void compute_o_x(
    internal_registers_format *internal_register,
    bfloat16_e8m7 *o_x[DIM]
);

#endif