#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#ifdef QUIK_WITH_CUSPARSELT
#include <cusparseLt.h>
#endif

namespace QUIK::matmul {
torch::Tensor v1_linear_a8_w8_b32_o32CUDA(const torch::Tensor &A,  // INT8
                                   const torch::Tensor &B, // INT8
                                   const torch::Tensor &bias    // INT32
);
torch::Tensor v2_linear_a8_w8_b32_o32CUDA(const torch::Tensor &A,  // INT8
                                   const torch::Tensor &B, // INT8
                                   const torch::Tensor &bias    // INT32
);
torch::Tensor v3_linear_a8_w8_b32_o32CUDA(const torch::Tensor &A,  // INT8
                                   const torch::Tensor &B, // INT8
                                   const torch::Tensor &bias    // INT32
);

torch::Tensor int8MatmulCUDA2(const torch::Tensor &A, const torch::Tensor &B);
torch::Tensor otherMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha);
torch::Tensor otherMatmul3CUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha);
torch::Tensor otherMatmulCUDA2(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &bias, const torch::Tensor &alpha);
std::vector<torch::Tensor> testCUDA(const torch::Tensor &input);
std::vector<torch::Tensor> testCUDA2(const torch::Tensor &input);
torch::Tensor linear_a8_w8_bfp32_ofp32CUDA(const torch::Tensor &A,  // INT8
                                   const torch::Tensor &B, // INT8
                                   const torch::Tensor &bias,    // INT32
                                   const torch::Tensor &alpha
);
std::vector<torch::Tensor> add_one_cuda(const torch::Tensor &input);
std::vector<torch::Tensor> add_forward(const torch::Tensor& input);

torch::Tensor Matmul_a_CUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha);
torch::Tensor Matmul_b_CUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha);
torch::Tensor Matmul_c_CUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha);

float accessor_get1CUDA(torch::Tensor a, int i);
float accessor_get2CUDA(torch::Tensor a, int i);
float accessor_get3CUDA(torch::Tensor a);



torch::Tensor int4MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);

torch::Tensor int4OutputInt8MatmulCUDA(const torch::Tensor &A,
                                       const torch::Tensor &B);

torch::Tensor int4SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E);

torch::Tensor int4OutputInt8SpMatmulCUDA(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &E);

torch::Tensor int4ReorderMeta(const torch::Tensor &E);

torch::Tensor int4GenRandomSparseMeta(int M, int K);

torch::Tensor int4Uncompress(const torch::Tensor &A, const torch::Tensor &E,
                             int M, int K);

torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);

torch::Tensor int8OutputInt8MatmulCUDA(const torch::Tensor &A,
                                       const torch::Tensor &B);

torch::Tensor int8SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E);

torch::Tensor int8OutputInt8SpMatmulCUDA(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &E);

torch::Tensor int8ReorderMeta(const torch::Tensor &E);

torch::Tensor int8GenRandomSparseMeta(int M, int K);

torch::Tensor int8Uncompress(const torch::Tensor &A, const torch::Tensor &E,
                             int M, int K);

#ifdef QUIK_WITH_CUSPARSELT

class CusparseLtInt8SpMatmul {
 private:
  using pytorchIndex = torch::IntArrayRef::value_type;
  const torch::Tensor &A_, &B_;
  torch::Tensor A_compressed_;
  int8_t *dA_ = nullptr;
  int8_t *dB_ = nullptr;
  int8_t *dA_compressed_ = nullptr;

  int64_t M_, N_, K_;

  cusparseOperation_t opA_, opB_;

  cusparseLtHandle_t *handle_;
  cusparseLtMatDescriptor_t *matA_, *matB_, *matC_;
  cusparseLtMatmulDescriptor_t *matmul_;
  cusparseLtMatmulAlgSelection_t *alg_sel_;
  cusparseLtMatmulPlan_t *plan_;
  cudaStream_t stream_ = nullptr;
  cudaStream_t *streams_ = nullptr;
  int num_streams_ = 0;

  float alpha_, beta_;

 public:
  CusparseLtInt8SpMatmul() = delete;
  CusparseLtInt8SpMatmul(const torch::Tensor &A, const torch::Tensor &B,
                         const int alg);
  ~CusparseLtInt8SpMatmul();

  void compress();
  torch::Tensor matmulBy(const torch::Tensor &B);
  torch::Tensor matmulDefault();
};

#endif
}  // namespace QUIK::matmul









#ifndef DEVICE_REGISTRY_HPP
#define DEVICE_REGISTRY_HPP

// Using <torch/extension.h> is recommended in the official documentation in
// https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-the-c-op.
// You can use <torch/types.h> for compatibility with CUDA 9.0
// Read https://github.com/pytorch/extension-cpp/issues/35 for more details.
#include <torch/extension.h>

#include <cassert>
#include <functional>
#include <map>
#include <type_traits>

inline std::string GetDeviceStr(const at::Device& device)
{
    std::string str = DeviceTypeName(device.type(), true);
    if (device.has_index()) {
        str.push_back(':');
        str.append(std::to_string(device.index()));
    }
    return str;
}

// Registry
template <typename F, F f>
class DeviceRegistry;

template <typename Ret, typename... Args, Ret (*f)(Args...)>
class DeviceRegistry<Ret (*)(Args...), f> {
public:
    using FunctionType = Ret (*)(Args...);
    static const int MAX_DEVICE_TYPES = int8_t(at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

    void Register(at::DeviceType device, FunctionType function)
    {
        funcs_[int8_t(device)] = function;
    }

    FunctionType Find(at::DeviceType device) const
    {
        return funcs_[int8_t(device)];
    }

    static DeviceRegistry& instance()
    {
        static DeviceRegistry inst;
        return inst;
    }

private:
    DeviceRegistry()
    {
        for (size_t i = 0; i < MAX_DEVICE_TYPES; ++i) {
            funcs_[i] = nullptr;
        }
    };
    FunctionType funcs_[MAX_DEVICE_TYPES];
};

// get device of first tensor param

template <typename T, typename... Args,
    std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value,
        bool>
    = true>
at::Device GetFirstTensorDevice(T&& t, Args&&... args)
{
    return std::forward<T>(t).device();
}
template <typename T, typename... Args,
    std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value,
        bool>
    = true>
at::Device GetFirstTensorDevice(T&& t, Args&&... args)
{
    return GetFirstTensorDevice(std::forward<Args>(args)...);
}

// check device consistency

inline std::pair<int, at::Device> CheckDeviceConsistency(
    const at::Device& device, int index)
{
    return { index, device };
}

template <typename T, typename... Args,
    std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value,
        bool>
    = true>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
    int index, T&& t,
    Args&&... args);

template <typename T, typename... Args,
    std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value,
        bool>
    = true>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
    int index, T&& t,
    Args&&... args)
{
    auto new_device = std::forward<T>(t).device();
    if (new_device.type() != device.type() || new_device.index() != device.index()) {
        return { index, new_device };
    }
    return CheckDeviceConsistency(device, index + 1, std::forward<Args>(args)...);
}

template <
    typename T, typename... Args,
    std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value, bool>>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
    int index, T&& t,
    Args&&... args)
{
    return CheckDeviceConsistency(device, index + 1, std::forward<Args>(args)...);
}

// dispatch

template <typename R, typename... Args>
auto Dispatch(const R& registry, const char* name, Args&&... args)
{
    auto device = GetFirstTensorDevice(std::forward<Args>(args)...);
    auto inconsist = CheckDeviceConsistency(device, 0, std::forward<Args>(args)...);
    TORCH_CHECK(inconsist.first >= int(sizeof...(Args)), name, ": at param ",
        inconsist.first,
        ", inconsistent device: ", GetDeviceStr(inconsist.second).c_str(),
        " vs ", GetDeviceStr(device).c_str(), "\n")
    auto f_ptr = registry.Find(device.type());
    TORCH_CHECK(f_ptr != nullptr, name, ": implementation for device ",
        GetDeviceStr(device).c_str(), " not found.\n")
    return f_ptr(std::forward<Args>(args)...);
}

// helper macro

#define DEVICE_REGISTRY(key) DeviceRegistry<decltype(&(key)), key>::instance()

#define REGISTER_DEVICE_IMPL(key, device, value)                 \
    struct key##_##device##_registerer {                         \
        key##_##device##_registerer()                            \
        {                                                        \
            DEVICE_REGISTRY(key).Register(at::k##device, value); \
        }                                                        \
    };                                                           \
    static key##_##device##_registerer _##key##_##device##_registerer;

#define DISPATCH_DEVICE_IMPL(key, ...) \
    Dispatch(DEVICE_REGISTRY(key), #key, __VA_ARGS__)

#endif // DEVICE_REGISTRY_HPP
