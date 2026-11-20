// USE_EXLLAMA=true, USE_V2_FORMAT=false, LEGACY_AWQ=false.
#define HGEMM_GPTQ_FORMAT_IMPLEMENTATION
#include "hgemm_gptq.h"

namespace hgemm_marlin_gptq {
#define HGEMM_DEFINE(T, W, BM, BN, BK, ZP, MP, NKP, S, V, A)   HGEMM_FORMAT_INSTANCE(template, T, W, BM, BN, BK, ZP, MP, NKP, S, V, A)
HGEMM_FORMAT_CONFIGS(HGEMM_DEFINE, true, false, false)
#undef HGEMM_DEFINE
}  // namespace hgemm_marlin_gptq
