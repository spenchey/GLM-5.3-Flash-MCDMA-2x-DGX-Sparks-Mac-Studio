#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void scalar_qmm_cuda(
    const at::Tensor& x,
    const at::Tensor& words,
    const at::Tensor& scales,
    const at::Tensor& biases,
    at::Tensor& out,
    int gs);

void scalar_qmm(
    const at::Tensor& x,
    const at::Tensor& words,
    const at::Tensor& scales,
    const at::Tensor& biases,
    at::Tensor& out,
    int64_t gs) {
  TORCH_CHECK(gs == 32 || gs == 64, "group size must be 32 or 64");
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
          x.stride(1) == 1,
      "x must be a CUDA bfloat16 matrix with contiguous rows");
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  TORCH_CHECK(k % gs == 0, "K must contain whole quantization groups");
  TORCH_CHECK(
      words.is_cuda() && words.is_contiguous() &&
          words.scalar_type() == at::kInt && words.dim() == 2 &&
          words.size(1) == k / 8,
      "words must be contiguous CUDA int32 Q4 rows");
  const int64_t n = words.size(0);
  TORCH_CHECK(
      scales.is_cuda() && scales.is_contiguous() &&
          scales.scalar_type() == at::kBFloat16 &&
          scales.sizes() == at::IntArrayRef({n, k / gs}),
      "scales must be contiguous CUDA bfloat16 (N, K / group)");
  TORCH_CHECK(
      biases.is_cuda() && biases.is_contiguous() &&
          biases.scalar_type() == at::kBFloat16 &&
          biases.sizes() == scales.sizes(),
      "biases must match scales");
  TORCH_CHECK(
      out.is_cuda() && out.is_contiguous() &&
          out.scalar_type() == at::kBFloat16 &&
          out.sizes() == at::IntArrayRef({m, n}),
      "out must be contiguous CUDA bfloat16 (M, N)");
  c10::cuda::CUDAGuard guard(x.device());
  scalar_qmm_cuda(x, words, scales, biases, out, static_cast<int>(gs));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("scalar_qmm", &scalar_qmm);
}
