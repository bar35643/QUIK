#include "matmul/matmul.h"
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <torch/extension.h>
#include <vector>

#include "matmul/matmul_internal.h"


using namespace at;

namespace QUIK::matmul {
torch::Tensor int4Matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("int4Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4Matmul", {A, B}, at::DeviceType::CUDA);
  return int4MatmulCUDA(A, B);
}

torch::Tensor int4OutputInt8Matmul(const torch::Tensor &A,
                                   const torch::Tensor &B) {
  torch::checkAllContiguous("int4OutputInt8Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4OutputInt8Matmul", {A, B}, at::DeviceType::CUDA);
  return int4OutputInt8MatmulCUDA(A, B);
}

torch::Tensor int4SpMatmul(const torch::Tensor &A, const torch::Tensor &B,
                           const torch::Tensor &E) {
  torch::checkAllContiguous("int4SpMatmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4SpMatmul", {A, B, E}, at::DeviceType::CUDA);
  return int4SpMatmulCUDA(A, B, E);
}

torch::Tensor int4OutputInt8SpMatmul(const torch::Tensor &A,
                                     const torch::Tensor &B,
                                     const torch::Tensor &E) {
  torch::checkAllContiguous("int4OutputInt8SpMatmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4OutputInt8SpMatmul", {A, B, E},
                         at::DeviceType::CUDA);
  return int4OutputInt8SpMatmulCUDA(A, B, E);
}

torch::Tensor int8Matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8Matmul", {A, B}, at::DeviceType::CUDA);
  return int8MatmulCUDA(A, B);
}

torch::Tensor int8OutputInt8Matmul(const torch::Tensor &A,
                                   const torch::Tensor &B) {
  torch::checkAllContiguous("int8OutputInt8Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8OutputInt8Matmul", {A, B}, at::DeviceType::CUDA);
  return int8OutputInt8MatmulCUDA(A, B);
}

torch::Tensor int8SpMatmul(const torch::Tensor &A, const torch::Tensor &B,
                           const torch::Tensor &E) {
  torch::checkAllContiguous("int8SpMatmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8SpMatmul", {A, B, E}, at::DeviceType::CUDA);
  return int8SpMatmulCUDA(A, B, E);
}

torch::Tensor int8OutputInt8SpMatmul(const torch::Tensor &A,
                                     const torch::Tensor &B,
                                     const torch::Tensor &E) {
  torch::checkAllContiguous("int8OutputInt8Matmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8OutputInt8Matmul", {A, B, E},
                         at::DeviceType::CUDA);
  return int8OutputInt8SpMatmulCUDA(A, B, E);
}








torch::Tensor v1_linear_a8_w8_b32_o32(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &bias) {
  torch::checkAllContiguous("v1_linear_a8_w8_b32_o32", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("v1_linear_a8_w8_b32_o32", {A, B, bias}, at::DeviceType::CUDA);
  return v1_linear_a8_w8_b32_o32CUDA(A, B, bias);
}

torch::Tensor v2_linear_a8_w8_b32_o32(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &bias) {
  torch::checkAllContiguous("v2_linear_a8_w8_b32_o32", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("v2_linear_a8_w8_b32_o32", {A, B, bias}, at::DeviceType::CUDA);
  return v2_linear_a8_w8_b32_o32CUDA(A, B, bias);
}

torch::Tensor v3_linear_a8_w8_b32_o32(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &bias) {
  torch::checkAllContiguous("v3_linear_a8_w8_b32_o32", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("v3_linear_a8_w8_b32_o32", {A, B, bias}, at::DeviceType::CUDA);
  return v3_linear_a8_w8_b32_o32CUDA(A, B, bias);
}

torch::Tensor linear_a8_w8_bfp32_ofp32(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &bias, const torch::Tensor &alpha) {
  torch::checkAllContiguous("linear_a8_w8_bfp32_ofp32", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}, {alpha, "alpha", 3}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("linear_a8_w8_bfp32_ofp32", {A, B, bias, alpha}, at::DeviceType::CUDA);
  return linear_a8_w8_bfp32_ofp32CUDA(A, B, bias, alpha);
}

torch::Tensor int8Matmul2(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("int8Matmul2", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8Matmul2", {A, B}, at::DeviceType::CUDA);
  return int8MatmulCUDA2(A, B);
}

torch::Tensor otherMatmul(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllContiguous("otherMatmul", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("otherMatmul", {A, B, alpha}, at::DeviceType::CUDA);
  return otherMatmulCUDA(A, B, alpha);
}
torch::Tensor otherMatmul2(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &bias, const torch::Tensor &alpha) {
  torch::checkAllContiguous("otherMatmul2", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}, {alpha, "alpha", 3}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("otherMatmul2", {A, B, bias, alpha}, at::DeviceType::CUDA);
  return otherMatmulCUDA2(A, B, bias, alpha);
}

torch::Tensor otherMatmul3(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllContiguous("otherMatmul3", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("otherMatmul3", {A, B, alpha}, at::DeviceType::CUDA);
  return otherMatmul3CUDA(A, B, alpha);
}

std::vector<torch::Tensor> test(const torch::Tensor &input) {
  //torch::checkAllContiguous("test", {{A, "A", 0}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("test", {input}, at::DeviceType::CUDA);
  return testCUDA(input);
}

std::vector<torch::Tensor> test2(const torch::Tensor &input) {
  //torch::checkAllContiguous("test", {{A, "A", 0}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("test2", {input}, at::DeviceType::CUDA);
  return testCUDA2(input);
}

std::vector<torch::Tensor> add_one(const torch::Tensor& input)
{
    return DISPATCH_DEVICE_IMPL(add_forward, input);
}


torch::Tensor Matmul_a(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllContiguous("Matmul_a", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  torch::checkDeviceType("Matmul_a", {A, B, alpha}, at::DeviceType::CUDA);
  return Matmul_a_CUDA(A, B, alpha);
}
torch::Tensor Matmul_b(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllContiguous("Matmul_b", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  torch::checkDeviceType("Matmul_b", {A, B, alpha}, at::DeviceType::CUDA);
  return Matmul_b_CUDA(A, B, alpha);
}
torch::Tensor Matmul_c(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllContiguous("Matmul_c", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  torch::checkDeviceType("Matmul_c", {A, B, alpha}, at::DeviceType::CUDA);
  return Matmul_c_CUDA(A, B, alpha);
}

float accessor_get1(torch::Tensor a, int i) {
  return accessor_get1CUDA(a, i);
}

float accessor_get2(torch::Tensor a, int i) {
  return accessor_get2CUDA(a, i);
}

float accessor_get3(torch::Tensor a) {
  return accessor_get3CUDA(a);
}

void buildSubmodule(py::module &mod) {
  py::module m = mod.def_submodule("matmul", "Matmul Functions");
  m.def("Matmul_c", &Matmul_c,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("alpha"));
  m.def("Matmul_b", &Matmul_b,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("alpha"));
  m.def("Matmul_a", &Matmul_a,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("alpha"));
  m.def("accessor_get1", &accessor_get1,"input: input", py::arg("a"), py::arg("i"));
  m.def("accessor_get2", &accessor_get2,"input: input", py::arg("a"), py::arg("i"));
  m.def("accessor_get3", &accessor_get3,"input: input", py::arg("a"));



  m.def("add_one", &add_one,"input: input", py::arg("input"));
  //m.impl("add_one", c10::DispatchKey::CUDA, TORCH_FN(add_one));


  m.def("test", &test,
        "input: input",
        py::arg("input"));

  m.def("test2", &test2,
        "input: input",
        py::arg("input"));

  m.def("linear_a8_w8_bfp32_ofp32", &linear_a8_w8_bfp32_ofp32, 
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, INT8, CUDA), bias: torch.Tensor(N, INT32, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("bias"), py::arg("alpha"));

  m.def("v1_linear_a8_w8_b32_o32", &v1_linear_a8_w8_b32_o32, 
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, INT8, CUDA), bias: torch.Tensor(N, INT32, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("bias"));

  m.def("v2_linear_a8_w8_b32_o32", &v2_linear_a8_w8_b32_o32, 
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, INT8, CUDA), bias: torch.Tensor(N, INT32, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("bias"));


  m.def("v3_linear_a8_w8_b32_o32", &v3_linear_a8_w8_b32_o32, 
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, INT8, CUDA), bias: torch.Tensor(N, INT32, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("bias"));

  m.def("int8Matmul2", &int8Matmul2,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"));

  m.def("otherMatmul", &otherMatmul,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("alpha"));
  m.def("otherMatmul3", &otherMatmul3,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("alpha"));

  m.def("otherMatmul2", &otherMatmul2,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"), py::arg("bias"), py::arg("alpha"));



  m.def("int4Matmul", &int4Matmul,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));

  m.def(
      "int4SpMatmul", &int4SpMatmul,
      "input: (A: torch.Tensor(M x K / 2 / 2, UINT8, CUDA), B: torch.Tensor(N "
      "x K / 2, UINT8, CUDA), E: torch.Tensor(M x K / 2 / 2, INT32, CUDA)\n"
      "output: torch.Tensor(M x N, INT32, CUDA)\n"
      "output = sp2dn(int4Unpacking(A), E) @ int4Unpacking(B)^T",
      py::arg("A"), py::arg("B"), py::arg("E"));

  m.def("int4ReorderMeta", &int4ReorderMeta,
        "input: (E: torch.Tensor(M x ?, INT32, CPU), N)\n"
        "output: torch.Tensor(INT32, CPU)\n"
        "output = cutlass::reorder_meta(E)",
        py::arg("E"));

  m.def("int4GenRandomSparseMeta", &int4GenRandomSparseMeta, "");
  m.def("int4Uncompress", &int4Uncompress, "");

  m.def("int8Matmul", &int8Matmul,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"));

  m.def("int8SpMatmul", &int8SpMatmul,
        "input: (A: torch.Tensor(M x K / 2, UINT8, CUDA), B: torch.Tensor(N x "
        "K / 2, UINT8, CUDA), E: torch.Tensor(M x K / 2 / 2, INT32, CUDA)\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = sp2dn(A, E) @ B^T",
        py::arg("A"), py::arg("B"), py::arg("E"));

  m.def("int8ReorderMeta", &int8ReorderMeta,
        "input: (E: torch.Tensor(M x ?, INT32, CPU), N)\n"
        "output: torch.Tensor(INT32, CPU)\n"
        "output = cutlass::reorder_meta(E)",
        py::arg("E"));

  m.def("int8GenRandomSparseMeta", &int8GenRandomSparseMeta, "");
  m.def("int8Uncompress", &int8Uncompress, "");

#ifdef QUIK_WITH_CUSPARSELT
  py::class_<CusparseLtInt8SpMatmul>(m, "CusparseLtInt8SpMatmul")
      .def(py::init<const torch::Tensor &, const torch::Tensor &, const int>(),
           "", py::arg("A"), py::arg("B"), py::arg("alg") = 0)
      .def("compress", &CusparseLtInt8SpMatmul::compress, "")
      .def("matmul_by", &CusparseLtInt8SpMatmul::matmulBy, "", py::arg("B"));
#endif
}
}  // namespace QUIK::matmul
