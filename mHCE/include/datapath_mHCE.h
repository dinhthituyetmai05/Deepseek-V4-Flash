#ifndef DATAPATH_MHCE_H
#define DATAPATH_MHCE_H

#include "load_mHCE.h"
#include "kernel_mHCE.h"
#include "store_mHCE.h"

void FirstLayer___Preblock_mHCE();

void FirstLayer___Residual_mixing_attn___Postblock_mHCE();

void PreviousLayer___Residual_mixing_attn___NextLayer___Preblock_mHCE();

void CurrentLayer___Residual_mixing_moe___Postblock_mHCE();

void LastLayer___Residual_mixing_moe();

#endif