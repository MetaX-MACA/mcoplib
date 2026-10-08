/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc2/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh
 * Copyright (c) 2026, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef USE_MACA
// 如果包含 cuda_bf16.h 后仍然报错，说明 MACA 没有自动 alias __nv_bfloat16
// 请添加以下 typedef 将 MACA 的 bfloat16 映射过来（常见为 __maca_bfloat16 或 maca_bfloat16）
#ifndef __nv_bfloat16
#define __nv_bfloat16 __maca_bfloat16 
#endif
#endif

namespace vllm {
namespace moe {
namespace reduce_topk {
namespace cg = cooperative_groups;
static constexpr int kWARP_SIZE = 32;

template <typename T_>
struct TopKRedType {
  using T = T_;
  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, half> ||
          std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, int>,
      "Top K reduction only implemented for int, float, float16 and bfloat16");

  using TypeCmp = std::conditional_t<sizeof(T) == 4, uint64_t, uint32_t>;
  using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int16_t>;

  static constexpr int kMoveBits = (sizeof(T) == 4) ? 32 : 16;
  static constexpr int kMaxIdx = 65535;
  TypeCmp compValIdx;

  static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0) {
    auto valueBits = cub::Traits<T>::TwiddleIn(
        reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
    TypeCmp compactTmp = valueBits;
    compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
    // Use 65535 minus idx to give higher priority to elements with smaller
    // indices.
    return compactTmp;
  }

  static __host__ __device__ void unpack(T& value, int32_t& index,
                                         TypeCmp cmp) {
    // Since “65535-idx” is always smaller than 65536 and positive, we can
    // directly use it as the lower 16 bits
    index = kMaxIdx - static_cast<int32_t>((cmp & 0xFFFF));

    auto compactTmp = cmp >> kMoveBits;
    auto valueBits = cub::Traits<T>::TwiddleOut(
        reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(compactTmp));
    value = reinterpret_cast<T&>(valueBits);
  }

  __host__ __device__ TopKRedType() = default;

  __host__ __device__ TopKRedType(T val, int32_t idx)
      : compValIdx(makeCmpVal(val, idx)) {}

  __host__ __device__ operator TypeCmp() const noexcept { return compValIdx; }

  __device__ inline TypeCmp reduce(
      cg::thread_block_tile<kWARP_SIZE> const& warp) {
    return cg::reduce(warp, compValIdx, cg::greater<TypeCmp>{});
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K_, bool Enable_>
struct TopKIdx {
  // by default, empty
};

template <int K_>
struct TopKIdx<K_, true> {
  static constexpr int K = K_;
  int32_t val[K];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TOPK_SWAP(I, J)                                         \
  {                                                             \
    auto pairMin = min(topK[I].compValIdx, topK[J].compValIdx); \
    auto pairMax = max(topK[I].compValIdx, topK[J].compValIdx); \
    topK[I].compValIdx = pairMax;                               \
    topK[J].compValIdx = pairMin;                               \
  }

template <int N, typename RedType>
struct Sort {
  // Generic descending sort (selection sort). Used for N without a
  // hand-optimized specialization below (e.g. N=22 for Nemotron top-22).
  static __device__ void run(RedType* topK) {
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
#pragma unroll
      for (int j = i + 1; j < N; ++j) {
        if (topK[i].compValIdx < topK[j].compValIdx) {
          RedType tmp = topK[i];
          topK[i] = topK[j];
          topK[j] = tmp;
        }
      }
    }
  }
};

template <typename RedType>
struct Sort<1, RedType> {
  static __device__ void run(RedType* topK) {}
};

template <typename RedType>
struct Sort<2, RedType> {
  static __device__ void run(RedType* topK) { TOPK_SWAP(0, 1); }
};

template <typename RedType>
struct Sort<3, RedType> {
  static __device__ void run(RedType* topK) {
    TOPK_SWAP(0, 1);
    TOPK_SWAP(1, 2);
    TOPK_SWAP(0, 1);
  }
};

template <typename RedType>
struct Sort<4, RedType> {
  static __device__ void run(RedType* topK) {
    TOPK_SWAP(0, 2);
    TOPK_SWAP(1, 3);
    TOPK_SWAP(0, 1);
    TOPK_SWAP(2, 3);
    TOPK_SWAP(1, 2);
  }
};

template <int K, typename Type>
__forceinline__ __device__ void reduceTopK(
    cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type value, int32_t idx, Type const minValue,
    int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
  using RedType = TopKRedType<Type>;
  RedType topK{value, idx};
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  for (int kk = 0; kk < actualK; ++kk) {
    topK =
        kk > 0 && packedMax == topK.compValIdx ? RedType{minValue, idx} : topK;
    // get the next largest value
    packedMax = topK.reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopKFunc(cg::thread_block_tile<kWARP_SIZE> const& warp,
                               Type (&out)[K], int32_t (&outIdx)[K],
                               Type (&value)[N], int32_t (&idx)[N],
                               Type minValue, int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(N < 5,
                "Only support candidates number less than or equal to 128");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};
  }

  if constexpr (!IsSorted) {
    Sort<N, RedType>::run(topK);
  }
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  for (int kk = 0; kk < actualK; ++kk) {
    bool update = kk > 0 && packedMax == topK[0].compValIdx;
#pragma unroll
    for (int nn = 0; nn < N; ++nn) {
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]}
                 : update              ? topK[nn + 1]
                                       : topK[nn];
    }
    // get the next largest value
    packedMax = topK[0].reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopK(
    cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type (&value)[N], int32_t (&idx)[N],
    Type const minValue, int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(
      N <= 16,
      "Only support candidates number less than or equal to 16*32=512");
  static_assert(N <= 4 || N % 4 == 0,
                "Only support candidates number is a multiple of 4*32=128 or "
                "less than or equal to 4");
  using RedType = TopKRedType<Type>;

  if constexpr (N <= 4) {
    reduceTopKFunc<K, Type, N>(warp, out, outIdx, value, idx, minValue,
                               actualK);
  } else {
    constexpr int numLoops = N / 4;
    constexpr int numResults = (numLoops * K - 1) / kWARP_SIZE + 1;

    Type topKBufferValue[numResults];
    int32_t topKBufferIdx[numResults];
    int32_t laneIdx = threadIdx.x % kWARP_SIZE;

    for (int ii = 0; ii < numResults; ++ii) {
      topKBufferValue[ii] = minValue;
      topKBufferIdx[ii] = ii * kWARP_SIZE - 1;
    }
    for (int loop = 0; loop < numLoops; ++loop) {
      int start = loop * 4;
      Type topKValue[K];
      int32_t topKIdx[K];
      Type inValue[4];
      int32_t inIdx[4];
      for (int i = 0; i < 4; ++i) {
        inValue[i] = value[start + i];
        inIdx[i] = idx[start + i];
      }
      reduceTopKFunc<K, Type, 4>(warp, topKValue, topKIdx, inValue, inIdx,
                                 minValue, actualK);
      int inOffset = laneIdx % K;
      if (laneIdx >= loop * K && laneIdx < (loop + 1) * K) {
        topKBufferValue[0] = topKValue[inOffset];
        topKBufferIdx[0] = topKIdx[inOffset];
      }
      if (loop == numLoops - 1 && (laneIdx < (numLoops * K - kWARP_SIZE))) {
        topKBufferValue[1] = topKValue[inOffset];
        topKBufferIdx[1] = topKIdx[inOffset];
      }
    }

    reduceTopKFunc<K, Type, numResults>(warp, out, outIdx, topKBufferValue,
                                        topKBufferIdx, minValue, actualK);
  }
};

template <int N>
struct IsPowerOf2 {
  static constexpr bool value = N > 0 && (N & (N - 1)) == 0;
};

template <int NumExperts, int NumTopExperts, int MinExperts, int MaxExperts,
          int MinTopExperts, int MaxTopExperts>
struct LaneOwnedTopKRange {
  static_assert(MinExperts > 0 && MinExperts <= MaxExperts);
  static_assert(MinTopExperts > 0 && MinTopExperts <= MaxTopExperts);
  static constexpr bool kEnabled =
      NumExperts >= MinExperts && NumExperts <= MaxExperts &&
      NumTopExperts >= MinTopExperts && NumTopExperts <= MaxTopExperts;
};

static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_EXPERTS = 512;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_EXPERTS = 1024;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_TOP_EXPERTS = 9;
static constexpr int kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_TOP_EXPERTS = 16;

template <int NumExperts, int NumTopExperts>
using HighExpertLaneOwnedTopKRange =
    LaneOwnedTopKRange<NumExperts, NumTopExperts,
                       kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_EXPERTS,
                       kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_EXPERTS,
                       kHIGH_EXPERT_LANE_OWNED_TOPK_MIN_TOP_EXPERTS,
                       kHIGH_EXPERT_LANE_OWNED_TOPK_MAX_TOP_EXPERTS>;

template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopKForLane(
    cg::thread_block_tile<kWARP_SIZE> const& warp, Type& out, int32_t& outIdx,
    Type (&value)[N], int32_t (&idx)[N], Type const minValue, int32_t laneIdx) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K <= kWARP_SIZE, "Top K must have K <= kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(N <= 64,
                "Only support candidates number less than or equal to "
                "64*32=2048");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};
  }

  Sort<N, RedType>::run(topK);

  typename RedType::TypeCmp packedMax{};
  typename RedType::TypeCmp lanePacked{};
#pragma unroll
  for (int kk = 0; kk < K; ++kk) {
    bool update = kk > 0 && packedMax == topK[0].compValIdx;
#pragma unroll
    for (int nn = 0; nn < N; ++nn) {
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]}
                 : update              ? topK[nn + 1]
                                       : topK[nn];
    }
    packedMax = topK[0].reduce(warp);
    if (laneIdx == kk) {
      lanePacked = packedMax;
    }
  }

  if (laneIdx < K) {
    RedType::unpack(out, outIdx, lanePacked);
  } else {
    out = minValue;
    outIdx = -1;
  }
}

// ---- bitonic merge top-K (generic: Chunks candidates/lane -> top-K) ----
// Packed value: [twiddled_value(32) | 0(16) | (65535-idx)(16)]; larger = better.

// Bitonic merge of a length-K bitonic sequence to descending order.
template <int K, typename RedType>
__device__ __forceinline__ void bmSort(RedType* X) {
#pragma unroll
  for (int off = K / 2; off >= 1; off >>= 1) {
#pragma unroll
    for (int i = 0; i < K; ++i) {
      int j = i ^ off;
      if (j > i && X[i].compValIdx < X[j].compValIdx) {
        RedType t = X[i]; X[i] = X[j]; X[j] = t;
      }
    }
  }
}

// Bitonic top-K merge of two sorted-descending K-lists A and B.
template <int K, typename RedType>
__device__ __forceinline__ void bmMerge(RedType* A, RedType* B) {
#pragma unroll
  for (int k = 0; k < K / 2; ++k) {
    RedType t = B[k];
    B[k] = B[K - 1 - k];
    B[K - 1 - k] = t;
  }
#pragma unroll
  for (int k = 0; k < K; ++k) {
    if (A[k].compValIdx < B[k].compValIdx) {
      RedType t = A[k];
      A[k] = B[k];
      B[k] = t;
    }
  }
  if constexpr (IsPowerOf2<K>::value) {
    bmSort<K>(A);
  } else {
    Sort<K, RedType>::run(A);
  }
}

// Warp-wide top-K of `Chunks` candidates per lane via bitonic merge tournament.
template <int Chunks, int K>
__device__ __forceinline__ void warpTopKBitonic(
    float (&value)[Chunks], int32_t (&idx)[Chunks],
    float (&out_score)[K], int32_t (&out_idx)[K], float minValue) {
  static_assert(K >= 1 && K <= kWARP_SIZE, "K must be in [1, 32]");
  static_assert(Chunks >= 1 && Chunks <= 64, "Chunks must be in [1, 64]");
  using RedType = TopKRedType<float>;
  RedType A[Chunks];
#pragma unroll
  for (int k = 0; k < Chunks; ++k) A[k] = RedType{value[k], idx[k]};
  Sort<Chunks, RedType>::run(A);

  constexpr int Keep = Chunks < K ? Chunks : K;
  RedType X[K];
#pragma unroll
  for (int k = 0; k < Keep; ++k) X[k] = A[k];
#pragma unroll
  for (int k = Keep; k < K; ++k) X[k] = RedType{minValue, RedType::kMaxIdx};

#pragma unroll
  for (int off = 1; off <= 16; off <<= 1) {
    RedType B[K];
#pragma unroll
    for (int k = 0; k < K; ++k)
      B[k].compValIdx = __shfl_xor_sync(0xffffffffu, X[k].compValIdx, off);
    bmMerge<K, RedType>(X, B);
  }
#pragma unroll
  for (int k = 0; k < K; ++k)
    X[k].compValIdx = __shfl_sync(0xffffffffu, X[k].compValIdx, 0);
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float v;
    int32_t i;
    RedType::unpack(v, i, X[k].compValIdx);
    out_score[k] = v;
    out_idx[k] = i;
  }
}

#undef TOPK_SWAP

}  // namespace reduce_topk
}  // namespace moe
}  // namespace vllm