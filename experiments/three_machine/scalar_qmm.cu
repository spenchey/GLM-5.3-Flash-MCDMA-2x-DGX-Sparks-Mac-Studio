#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__global__ void scalar_qmm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ words,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ biases,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int K,
    int ldx,
    int gs) {
  const int64_t output =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = static_cast<int64_t>(M) * N;
  if (output >= count) return;
  const int row = static_cast<int>(output / N);
  const int col = static_cast<int>(output % N);
  const int packed_k = K / 8;
  const int groups = K / gs;
  float accumulator = 0.0f;
  for (int group = 0; group < groups; ++group) {
    const float scale = __bfloat162float(scales[col * groups + group]);
    const float bias = __bfloat162float(biases[col * groups + group]);
    const int packed_start = group * (gs / 8);
#pragma unroll 1
    for (int packed = 0; packed < gs / 8; ++packed) {
      const uint32_t word = static_cast<uint32_t>(
          words[col * packed_k + packed_start + packed]);
#pragma unroll
      for (int nibble = 0; nibble < 8; ++nibble) {
        const int k = group * gs + packed * 8 + nibble;
        const float quantized = static_cast<float>((word >> (4 * nibble)) & 0xF);
        const __nv_bfloat16 weight = __float2bfloat16_rn(
            __fmaf_rn(quantized, scale, bias));
        accumulator = __fmaf_rn(
            __bfloat162float(x[static_cast<int64_t>(row) * ldx + k]),
            __bfloat162float(weight),
            accumulator);
      }
    }
  }
  out[output] = __float2bfloat16_rn(accumulator);
}

}  // namespace

void scalar_qmm_cuda(
    const at::Tensor& x,
    const at::Tensor& words,
    const at::Tensor& scales,
    const at::Tensor& biases,
    at::Tensor& out,
    int gs) {
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(words.size(0));
  constexpr int threads = 256;
  const int64_t count = static_cast<int64_t>(M) * N;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  scalar_qmm_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const int32_t*>(words.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      M,
      N,
      K,
      static_cast<int>(x.stride(0)),
      gs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
