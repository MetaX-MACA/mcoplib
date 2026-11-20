// Shared existing dispatch axes for the five format translation units.
#pragma once

#define HGEMM_FORMAT_PREDS(X, T, W, BM, BN, BK, ZP, S, V, A) \
  X(T, W, BM, BN, BK, ZP, true, true, S, V, A) \
  X(T, W, BM, BN, BK, ZP, false, true, S, V, A) \
  X(T, W, BM, BN, BK, ZP, true, false, S, V, A) \
  X(T, W, BM, BN, BK, ZP, false, false, S, V, A)

#define HGEMM_FORMAT_K(X, T, W, BM, BN, ZP, S, V, A) \
  HGEMM_FORMAT_PREDS(X, T, W, BM, BN, 1, ZP, S, V, A) \
  HGEMM_FORMAT_PREDS(X, T, W, BM, BN, 2, ZP, S, V, A) \
  HGEMM_FORMAT_PREDS(X, T, W, BM, BN, 4, ZP, S, V, A) \
  HGEMM_FORMAT_PREDS(X, T, W, BM, BN, 8, ZP, S, V, A)

#define HGEMM_FORMAT_TILES(X, T, W, S, V, A) \
  HGEMM_FORMAT_K(X, T, W, 1, 16, true, S, V, A) \
  HGEMM_FORMAT_K(X, T, W, 2, 16, true, S, V, A) \
  HGEMM_FORMAT_K(X, T, W, 3, 8, true, S, V, A) \
  HGEMM_FORMAT_K(X, T, W, 4, 8, true, S, V, A) \
  HGEMM_FORMAT_K(X, T, W, 8, 8, true, S, V, A)

#define HGEMM_FORMAT_CONFIGS(X, S, V, A) \
  HGEMM_FORMAT_TILES(X, half, kU4, S, V, A) \
  HGEMM_FORMAT_TILES(X, half, kU8, S, V, A) \
  HGEMM_FORMAT_TILES(X, __maca_bfloat16, kU4, S, V, A) \
  HGEMM_FORMAT_TILES(X, __maca_bfloat16, kU8, S, V, A)

// Legacy is reachable through both the GPTQ format facade and AWQ overload.
#define HGEMM_LEGACY_CONFIGS(X) \
  HGEMM_FORMAT_CONFIGS(X, true, true, true) \
  HGEMM_FORMAT_K(X, half, kU4, 1, 8, true, true, true, true) \
  HGEMM_FORMAT_K(X, half, kU4, 2, 8, true, true, true, true) \
  HGEMM_FORMAT_K(X, half, kU4, 1, 8, false, true, true, true) \
  HGEMM_FORMAT_K(X, half, kU4, 2, 8, false, true, true, true) \
  HGEMM_FORMAT_K(X, half, kU4, 3, 8, false, true, true, true) \
  HGEMM_FORMAT_K(X, half, kU4, 4, 8, false, true, true, true) \
  HGEMM_FORMAT_K(X, half, kU4, 1, 16, false, true, true, true) \
  HGEMM_FORMAT_K(X, half, kU4, 2, 16, false, true, true, true) \
  HGEMM_FORMAT_K(X, __maca_bfloat16, kU4, 1, 8, true, true, true, true) \
  HGEMM_FORMAT_K(X, __maca_bfloat16, kU4, 2, 8, true, true, true, true) \
  HGEMM_FORMAT_K(X, __maca_bfloat16, kU4, 1, 8, false, true, true, true) \
  HGEMM_FORMAT_K(X, __maca_bfloat16, kU4, 2, 8, false, true, true, true) \
  HGEMM_FORMAT_K(X, __maca_bfloat16, kU4, 3, 8, false, true, true, true) \
  HGEMM_FORMAT_K(X, __maca_bfloat16, kU4, 4, 8, false, true, true, true) \
  HGEMM_FORMAT_K(X, __maca_bfloat16, kU4, 1, 16, false, true, true, true) \
  HGEMM_FORMAT_K(X, __maca_bfloat16, kU4, 2, 16, false, true, true, true)

#define HGEMM_FORMAT_INSTANCE(STORAGE, T, W, BM, BN, BK, ZP, MP, NKP, S, V, A) \
  STORAGE bool launch_gemm_gptq_kernel_impl<T, vllm::W.id(), 256, BM, BN, BK, \
      false, ZP, MP, NKP, S, V, A>( \
      const PackTypeInt4*, const PackTypeInt4*, PackTypeInt4*, PackTypeInt4*, \
      const PackTypeInt4*, const PackTypeInt4*, int*, int, int, int, int, int, \
      cudaStream_t, bool, bool, bool);

