#ifndef CONFIG_MHCE_H
#define CONFIG_MHCE_H

#define TOKEN 1024
#define N_HC 4
#define DIM 4096

#define B 4

#define SK_ROUND 20



//Tạm thời load 1 phát hết weight luôn
#define WEIGHT_TILE_ROW N_HC*DIM

#define WEIGHT_PRE_TILE_COL N_HC
//#define WEIGHT_PRE_TILE_CHUNK 1 // số lượng PE chạy song song
//#define WEIGHT_PRE_TILE_ITER (N_HC*DIM/WEIGHT_PRE_TILE_CHUNK) // Số lần gọi PE

#define WEIGHT_RES_TILE_COL N_HC*N_HC
//#define WEIGHT_RES_TILE_CHUNK 1 // số lượng PE chạy song song
//#define WEIGHT_RES_TILE_ITER (N_HC*DIM/WEIGHT_RES_TILE_CHUNK) // Số lần gọi PE

#define WEIGHT_POST_TILE_COL N_HC
//#define WEIGHT_POST_TILE_CHUNK 1 // số lượng PE chạy song song
//#define WEIGHT_POST_TILE_ITER (N_HC*DIM/WEIGHT_POST_TILE_CHUNK) // Số lần gọi PE

#endif