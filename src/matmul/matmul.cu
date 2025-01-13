#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include "cutlass/gemm/device/gemm_universal.h"
#include <torch/cuda.h>

#include "int4.h"
#include "matmul/matmul_internal.h"
#include "util.h"
#include <vector>
#include <tuple>
#include <iostream>

#ifdef QUIK_WITH_CUSPARSELT
#include <cusparseLt.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/layout/permute.h"

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/CUDALoops.cuh>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <cuda.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <c10/util/BFloat16.h>
#include <ATen/native/DispatchStub.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/ATen.h>
#include <c10/util/TypeCast.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_numbers.h>
#include <algorithm>
#include <ATen/cuda/Atomic.cuh>
using namespace at;

const at::Tensor zero_value = torch::tensor(0.0f).cuda();
const at::Tensor one_value = torch::tensor(1.0f).cuda();

template<typename A, typename B> class cast_A_to_B_ptr {
    //Usage: >>> short d1 = 55;
    //       >>> cast_A_to_B_ptr<short, int> f1 { d1 };
    //       >>> std::cout << "RESULT : " << *f1 << std::endl;
    private:
      A& d;
    public:
      cast_A_to_B_ptr(A& d) : d(d) { }
      B operator*() const { return static_cast<B>(d); }
};



namespace QUIK::matmul {

float accessor_get1CUDA(torch::Tensor a, int i) {
    //a is 2-dimensional and holds floats.
    auto a_accessor = a.accessor<float, 2>();
    return a_accessor[i][i];
}

__global__ void packed_accessor_kernel(torch::PackedTensorAccessor64<float, 2> foo, float* trace) {
  int i = threadIdx.x;
  gpuAtomicAdd(trace, foo[i][i]);
}
//__device__
float accessor_get2CUDA(torch::Tensor a, int i) {
    //auto foo_a = a.packed_accessor64<float,2>();
    //float trace = 0;
    //packed_accessor_kernel<<<1, 3>>>(foo_a, &trace);
    //return trace;

    // //torch::Tensor foo = torch::rand({12, 12}, torch::dtype(torch::kF32).device(a.device())).contiguous();
    // auto foo_a = a.packed_accessor64<float,2>();
    // float trace = 0;
    // packed_accessor_kernel<<<1, 3>>>(foo_a, &trace);
    // torch::cuda::synchronize();
    // //at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // //AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    // return trace;

    //torch::Tensor ten = torch::rand({12, 12}, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)); 
    //std::vector<float> v(ten.data_ptr<float>(), ten.data_ptr<float>() + ten.numel());
    //return v[i];

    //return (float *)(a[i][i].data_ptr<float>())

    float tmp = -1;
    cudaMemcpy(&tmp, a[i][i].data_ptr<float>(), sizeof(float), cudaMemcpyDefault);
    //torch::cuda::synchronize();
    return tmp;


    //auto a_accessor = a.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
    //return a_accessor[i][i];
}

float accessor_get3CUDA(torch::Tensor a) {
    float tmp = 0;
    if(a.dtype() == torch::kBFloat16){
        cudaMemcpyAsync(reinterpret_cast<unsigned char *>(&tmp)+2, a.data_ptr(), 2, cudaMemcpyDefault);
    }else{
        cudaMemcpyAsync(&tmp, a.data_ptr<float>(), sizeof(float), cudaMemcpyDefault);
    }
    //__syncthreads();
    return tmp;
}




int64_t integer_round(int64_t num, int64_t denom){
  return (num + denom - 1) / denom;
}

template<class T>
__global__ void add_one_kernel(const T *const input, int8_t *const output, const int64_t N, const T *const scale){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //if (idx < N) output[idx] = static_cast<int8_t>(std::rint(input[idx] / *scale));
  //if (idx < N) output[idx] = static_cast<int8_t>(std::clamp(   (T)std::rint(input[idx] / *scale)   , (T)-127.0, (T)127.0)   );
  if (idx < N) output[idx] = static_cast<int8_t>(std::clamp(   (T)std::rint(input[idx] / *scale)   , (T)-127.0, (T)127.0)   );
}


std::vector<torch::Tensor> add_one_cuda(const torch::Tensor &input){
  auto output = torch::empty_like(input, torch::dtype(torch::kInt8)); //torch::zeros_like(input)
  const auto [min, max] = torch::aminmax(input);
  auto iter = TensorIteratorConfig().add_output(min)
                                    .add_input(max)
                                    .add_input(min)
                                    .resize_outputs(false)
                                    .check_all_same_dtype(false)
                                    .check_all_same_device(false)
                                    .build();
  //at::native::gpu_kernel(iter, []GPU_LAMBDA(float a, float b) -> float {return (a-b) / 254;});          
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,ScalarType::BFloat16,iter.common_dtype(),
      "add_one_cuda_1",[&]() {
        at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {return (a-b) / 254;});
      }
  );
  const auto scale = min; //const auto scale = (max - min) / 254;  

  // auto min_result = torch::empty_like(min);
  // auto max_result = torch::empty_like(min);
  // auto iter_tst = at::native::make_reduction("aminmax_cuda", min_result, max_result, input, input.dim(), false, input.scalar_type());
  // at::native::gpu_reduce_kernel<float, float, 1>(
  //   iter_tst,
  //   at::native::MinMaxOps<float, float, int32_t>{},
  //   thrust::pair<float, float>(at::numeric_limits<float>::upper_bound(), at::numeric_limits<float>::lower_bound())
  // );


  // auto iter2 = TensorIteratorConfig().add_output(output)
  //                                   .add_input(input)
  //                                   .add_input(scale)
  //                                   .resize_outputs(false)
  //                                   .check_all_same_dtype(false)
  //                                   .check_all_same_device(false)
  //                                   .build();
  // at::native::gpu_kernel(iter2, []GPU_LAMBDA(float a, float b) -> int8_t {return static_cast<int8_t>(std::rint(a / b));});  

  // Common values: AT_DISPATCH_INDEX_TYPES AT_DISPATCH_FLOATING_TYPES AT_DISPATCH_INTEGRAL_TYPES
  auto numel = input.numel();
   AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,ScalarType::BFloat16,input.scalar_type(), 
      "add_one_cuda_2", [&](){
      //FIXME: Corrupted Kernel
      //const auto block_size = 128;  //128
      //const auto num_blocks = std::min(65535L, integer_round(input.numel(), block_size));
      //add_one_kernel<<<num_blocks, block_size>>>(input.data_ptr<scalar_t>(),output.data_ptr<int8_t>(),input.numel(),scale.data_ptr<scalar_t>());

      //Working Kernel
      add_one_kernel<<<(numel+255)/256, 256>>>(input.data_ptr<scalar_t>(),output.data_ptr<int8_t>(),numel,scale.data_ptr<scalar_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK(); // Always test your kernel launches
    }
  );
  // AT_DISPATCH_ALL_TYPES(input.scalar_type(), 
  //     "add_one_cuda_2", [&](){
  //     const auto block_size = 128;  //128
  //     const auto num_blocks = std::min(65535L, integer_round(input.numel(), block_size));
  //     add_one_kernel<<<num_blocks, block_size>>>(input.data_ptr<scalar_t>(),output.data_ptr<int8_t>(),input.numel(),scale.data_ptr<scalar_t>());
  //     C10_CUDA_KERNEL_LAUNCH_CHECK(); // Always test your kernel launches
  //   }
  // );
  return {scale, output};
}
REGISTER_DEVICE_IMPL(add_forward, CUDA, add_one_cuda);

torch::Tensor otherMatmulCUDA2(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &bias, const torch::Tensor &alpha) {
  torch::checkAllSameGPU("otherMatmul2", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}, {alpha, "alpha", 3}});
  auto M = A.size(0);auto N = B.size(0);auto K = A.size(1);

  auto TorchDtype = torch::kBFloat16;//torch::kF32, torch::kBFloat16
  using ElementOutput = cutlass::bfloat16_t; //float, cutlass::bfloat16_t
  using ElementComputeEpilogue = float; //float
  using ElementAccumulator = int32_t;

  using Sm80Defaults = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, int8_t, int8_t, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,  cutlass::layout::RowMajor, int8_t,  cutlass::layout::ColumnMajor, //ElementA, LayoutA, ElementB, LayoutB
      ElementOutput, cutlass::layout::RowMajor,//ElementOutput, LayoutOutput
      ElementAccumulator,//ElementAccumulator
      cutlass::arch::OpClassTensorOp,//indicating Tensor Cores
      cutlass::arch::Sm80,//target GPU compute architecture
      Sm80Defaults::ThreadblockShape, Sm80Defaults::WarpShape, Sm80Defaults::InstructionShape,//ThreadblockShape, WarpShape, InstructionShape
      //cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//Swizzling function
      5, //Number of pipeline stages
      //Sm80Defaults::kAlignmentA, Sm80Defaults::kAlignmentB,/// Access granularity of matrix in units of elements
      4, 4,
      false,
      Sm80Defaults::Operator//cutlass::arch::OpMultiplyAddSaturate cutlass::arch::OpMultiplyAdd
      >;

  float* tmp1 = (alpha.to(torch::kF32, true)).data_ptr<float>();
  float* tmp2 = one_value.data_ptr<float>();
  auto C = torch::empty({M, N}, torch::dtype(TorchDtype).device(A.device()));
  Gemm gemmOp;
  typename Gemm::Arguments arguments{
      {static_cast<cutlass::gemm::GemmCoord::Index>(M), static_cast<cutlass::gemm::GemmCoord::Index>(N), static_cast<cutlass::gemm::GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {reinterpret_cast<ElementOutput *>(bias.to(TorchDtype).data_ptr()), 0},
      {reinterpret_cast<ElementOutput *>(   C.to(TorchDtype).data_ptr()), N},
      {tmp1, tmp2}}; //alpha.to(torch::kF32).item<float>()
  auto status = gemmOp(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,cutlassGetStatusString(status))
  return C; //C.to(torch::kBFloat16);
}

torch::Tensor otherMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllSameGPU("otherMatmul", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  auto M = A.size(0);auto N = B.size(0);auto K = A.size(1);

  auto TorchDtype = torch::kBFloat16;//torch::kF32, torch::kBFloat16
  using ElementOutput = cutlass::bfloat16_t; //float, cutlass::bfloat16_t
  using ElementComputeEpilogue = float; //float
  using ElementAccumulator = int32_t;

  using Sm80Defaults = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, int8_t, int8_t, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,  cutlass::layout::RowMajor, int8_t,  cutlass::layout::ColumnMajor, //ElementA, LayoutA, ElementB, LayoutB
      ElementOutput, cutlass::layout::RowMajor,//ElementOutput, LayoutOutput
      ElementAccumulator,//ElementAccumulator
      cutlass::arch::OpClassTensorOp,//indicating Tensor Cores
      cutlass::arch::Sm80,//target GPU compute architecture
      Sm80Defaults::ThreadblockShape, Sm80Defaults::WarpShape, Sm80Defaults::InstructionShape,//ThreadblockShape, WarpShape, InstructionShape
      //cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//Swizzling function
      5, //Number of pipeline stages
      //Sm80Defaults::kAlignmentA, Sm80Defaults::kAlignmentB,/// Access granularity of matrix in units of elements
      4, 4,
      false,
      Sm80Defaults::Operator//cutlass::arch::OpMultiplyAddSaturate cutlass::arch::OpMultiplyAdd
      >;

  float* tmp = (alpha.to(torch::kF32, true)).data_ptr<float>();
  auto C = torch::empty({M, N}, torch::dtype(TorchDtype).device(A.device()));
  Gemm gemmOp;
  typename Gemm::Arguments arguments{
      {static_cast<cutlass::gemm::GemmCoord::Index>(M), static_cast<cutlass::gemm::GemmCoord::Index>(N), static_cast<cutlass::gemm::GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {reinterpret_cast<ElementOutput *>(   C.to(TorchDtype).data_ptr()), N},
      {reinterpret_cast<ElementOutput *>(   C.to(TorchDtype).data_ptr()), N},
      {tmp}}; //alpha.to(torch::kF32).item<float>()
  auto status = gemmOp(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,cutlassGetStatusString(status))
  return C; //C.to(torch::kBFloat16);
}











std::vector<torch::Tensor> testCUDA2(const torch::Tensor &input) {
  const auto [min, max] = torch::aminmax(input);
  const auto scale = (max - min) / 254;
  auto out = torch::round(input / scale);//auto out = torch::floor_divide(input, scale);
  return {scale, out.to(torch::kInt8)};
}

std::vector<torch::Tensor> testCUDA(const torch::Tensor &input) {
  auto scale = (torch::max(input) - torch::min(input)) / 254;
  auto out = torch::round(input / scale).to(torch::kInt8);
  return {scale, out};
}

//cd QUIK/third-party/cutlass/ && git add . && git commit -m "tst" && cd ~
torch::Tensor otherMatmul3CUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllSameGPU("otherMatmul3", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  auto M = A.size(0);auto N = B.size(0);auto K = A.size(1);

  auto TorchDtype = torch::kBFloat16;//torch::kF32, torch::kBFloat16
  using ElementOutput = cutlass::bfloat16_t; //float, cutlass::bfloat16_t
  using ElementComputeEpilogue = float; //float
  using ElementAccumulator = int32_t;

  using Sm80Defaults = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, int8_t, int8_t, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,  cutlass::layout::RowMajor, int8_t,  cutlass::layout::ColumnMajor, //ElementA, LayoutA, ElementB, LayoutB
      ElementOutput, cutlass::layout::ColumnMajor,//ElementOutput, LayoutOutput
      ElementAccumulator,//ElementAccumulator
      cutlass::arch::OpClassTensorOp,//indicating Tensor Cores
      cutlass::arch::Sm80,//target GPU compute architecture
      Sm80Defaults::ThreadblockShape, Sm80Defaults::WarpShape, Sm80Defaults::InstructionShape,//ThreadblockShape, WarpShape, InstructionShape
      //cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//Swizzling function
      3, //Number of pipeline stages
      Sm80Defaults::kAlignmentA, Sm80Defaults::kAlignmentB,/// Access granularity of matrix in units of elements
      //4, 4,
      false,
      Sm80Defaults::Operator//cutlass::arch::OpMultiplyAddSaturate cutlass::arch::OpMultiplyAdd
      >;

  auto C = torch::empty({N, M}, torch::dtype(TorchDtype).device(A.device()));
  cutlass::TensorRef<int8_t,        cutlass::layout::RowMajor>    A_ref(                           A.data_ptr<int8_t>(),      cutlass::layout::RowMajor::packed(cutlass::MatrixCoord(M, K))); //M, K
  cutlass::TensorRef<int8_t,        cutlass::layout::ColumnMajor> B_ref(                           B.data_ptr<int8_t>(),   cutlass::layout::ColumnMajor::packed(cutlass::MatrixCoord(N, K))); //N, K
  cutlass::TensorRef<ElementOutput, cutlass::layout::ColumnMajor>  out_ref(reinterpret_cast<ElementOutput *>(C.data_ptr()),cutlass::layout::ColumnMajor::packed(cutlass::MatrixCoord(N, M))); //M, N

  Gemm gemmOp;
  typename Gemm::Arguments arguments{{static_cast<cutlass::gemm::GemmCoord::Index>(M), static_cast<cutlass::gemm::GemmCoord::Index>(N), static_cast<cutlass::gemm::GemmCoord::Index>(K)},
  A_ref,B_ref,out_ref,out_ref,{alpha.to(torch::kF32).item<float>(), 0}};


  auto status = gemmOp(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,cutlassGetStatusString(status))
  return C; //C.to(torch::kBFloat16);
}



























torch::Tensor Matmul_a_CUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllSameGPU("Matmul_a", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  auto M = A.size(0);auto N = B.size(0);auto K = A.size(1);

  using ElementOutput = float; //float, cutlass::bfloat16_t
  using Sm80Defaults = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, int8_t, int8_t, ElementOutput, int32_t>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,  cutlass::layout::RowMajor, int8_t,  cutlass::layout::ColumnMajor, //ElementA, LayoutA, ElementB, LayoutB
      ElementOutput, cutlass::layout::RowMajor,//ElementOutput, LayoutOutput
      int32_t,//ElementAccumulator
      cutlass::arch::OpClassTensorOp,//indicating Tensor Cores
      cutlass::arch::Sm80,//target GPU compute architecture
      Sm80Defaults::ThreadblockShape, Sm80Defaults::WarpShape, Sm80Defaults::InstructionShape,//ThreadblockShape, WarpShape, InstructionShape
      //cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, int32_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,//Swizzling function
      3, //Number of pipeline stages
      Sm80Defaults::kAlignmentA, Sm80Defaults::kAlignmentB,/// Access granularity of matrix in units of elements
      false,
      Sm80Defaults::Operator//cutlass::arch::OpMultiplyAddSaturate cutlass::arch::OpMultiplyAdd
      >;

  auto C = torch::empty({M, N}, torch::dtype(torch::kF32).device(A.device())); //kF32

  cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> out_ref(C.data_ptr<ElementOutput>(), cutlass::layout::RowMajor::packed(cutlass::MatrixCoord(M, N)));

  Gemm gemmOp;
  typename Gemm::Arguments arguments{
      {static_cast<cutlass::gemm::GemmCoord::Index>(M), static_cast<cutlass::gemm::GemmCoord::Index>(N), static_cast<cutlass::gemm::GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      out_ref,
      out_ref,
      {1, 0}};
  auto status = gemmOp(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,cutlassGetStatusString(status))
  return C; //return C.to(torch::kBFloat16);
}



torch::Tensor Matmul_b_CUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllSameGPU("Matmul_b", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  auto M = A.size(0); auto N = B.size(0); auto K = A.size(1); //2048

  using Sm80Defaults = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, int8_t, int8_t, float, int32_t>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,  cutlass::layout::RowMajor, int8_t,  cutlass::layout::ColumnMajor, 
      float,   cutlass::layout::RowMajor,//ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutOutput
      int32_t,//ElementAccumulator
      cutlass::arch::OpClassTensorOp,//indicating Tensor Cores
      cutlass::arch::Sm80,//target GPU compute architecture
      cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<float, 128 / cutlass::sizeof_bits<float>::value, int32_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,//Swizzling function
      3, Sm80Defaults::kAlignmentA, Sm80Defaults::kAlignmentB,//Number of pipeline stages,  Access granularity of matrix in units of elements
      false, Sm80Defaults::Operator//cutlass::arch::OpMultiplyAddSaturate cutlass::arch::OpMultiplyAdd
      >;

  auto C = torch::empty({M, N}, torch::dtype(torch::kF32).device(A.device())); //kF32
  cutlass::TensorRef<float, cutlass::layout::RowMajor> out_ref(C.data_ptr<float>(), cutlass::layout::RowMajor::packed(cutlass::MatrixCoord(M, N)));
  Gemm gemmOp;
  typename Gemm::Arguments arguments{
      {static_cast<cutlass::gemm::GemmCoord::Index>(M), static_cast<cutlass::gemm::GemmCoord::Index>(N), static_cast<cutlass::gemm::GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      out_ref,
      out_ref,
      {1, 0}};
  auto status = gemmOp(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,cutlassGetStatusString(status))
  return C; //return C.to(torch::kBFloat16);
}






torch::Tensor Matmul_c_CUDA(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alpha) {
  torch::checkAllSameGPU("Matmul_c", {{A, "A", 0}, {B, "B", 1}, {alpha, "alpha", 2}});
  auto M = A.size(0); auto N = B.size(0); auto K = A.size(1); //2048

  using Sm80Defaults = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, int8_t, int8_t, float, int32_t>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,  cutlass::layout::RowMajor, int8_t,  cutlass::layout::ColumnMajor, 
      float,   cutlass::layout::ColumnMajor,//ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutOutput
      int32_t,//ElementAccumulator
      cutlass::arch::OpClassTensorOp,//indicating Tensor Cores
      cutlass::arch::Sm80,//target GPU compute architecture
      cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<float, 128 / cutlass::sizeof_bits<float>::value, int32_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,//Swizzling function
      3, Sm80Defaults::kAlignmentA, Sm80Defaults::kAlignmentB,//Number of pipeline stages,  Access granularity of matrix in units of elements
      false, Sm80Defaults::Operator//cutlass::arch::OpMultiplyAddSaturate cutlass::arch::OpMultiplyAdd
      >;

  auto C = torch::empty({N, M}, torch::dtype(torch::kF32).device(A.device())); //kF32
  cutlass::TensorRef<float, cutlass::layout::ColumnMajor> out_ref(C.data_ptr<float>(), cutlass::layout::ColumnMajor::packed(cutlass::MatrixCoord(N, M)));
  Gemm gemmOp;
  typename Gemm::Arguments arguments{
      {static_cast<cutlass::gemm::GemmCoord::Index>(M), static_cast<cutlass::gemm::GemmCoord::Index>(N), static_cast<cutlass::gemm::GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      out_ref,
      out_ref,
      {1, 0}};
  auto status = gemmOp(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,cutlassGetStatusString(status))
  return C; //return C.to(torch::kBFloat16);
}



























































// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w8_bfp32_ofp32CUDA(const torch::Tensor &input,  // INT8
                                       const torch::Tensor &weight, // INT8
                                       const torch::Tensor &bias,   // FP32
                                       const torch::Tensor &alpha   // FP32
) {

  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);
  

  using ElementOutput = float;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;


  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, 
      int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, 
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp, 
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, 
      cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>
      >;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(out.data_ptr<ElementOutput>(), LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha.item<float>(), 0}, 1};
  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments); // Using the arguments, query for extra workspace required for matrix multiplication computation
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size); // Allocate workspace memory

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {throw std::runtime_error("cutlass cannot implement");}

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {throw std::runtime_error("cutlass cannot initialize");}

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {throw std::runtime_error("cutlass cannot run");}
  return out;//.toType(torch::kBFloat16);
}


torch::Tensor int8MatmulCUDA2(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("int8Matmul2", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>; // cutlass::gemm::GemmShape<256, 128, 64>,
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;          // cutlass::gemm::GemmShape<64, 64, 64>, 
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;     // cutlass::gemm::GemmShape<16, 8, 32>,

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,  cutlass::layout::RowMajor,    // ElementA, LayoutA
      int8_t,  cutlass::layout::ColumnMajor, // ElementB, LayoutB
      int32_t, cutlass::layout::RowMajor,    // ElementOutput, LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80,  // tag indicating target GPU compute architecture
      ThreadblockShape,
      WarpShape,
      InstructionShape
      >;

  Gemm gemmOp;
  using GemmCoord = cutlass::gemm::GemmCoord;
  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {C.data_ptr<int32_t>(), N},
      {C.data_ptr<int32_t>(), N},
      {1, 0}};
  auto status = gemmOp(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,cutlassGetStatusString(status))
  return C;
}

torch::Tensor v3_linear_a8_w8_b32_o32CUDA(const torch::Tensor &A,  // INT8
                                   const torch::Tensor &B, // INT8
                                   const torch::Tensor &bias    // INT32
) {
  torch::checkAllSameGPU("linear_a8_w8_b32_o32", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = int32_t;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>; // cutlass::gemm::GemmShape<256, 128, 64>,
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;          // cutlass::gemm::GemmShape<64, 64, 64>, 
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;     // cutlass::gemm::GemmShape<16, 8, 32>,

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, 
      cutlass::layout::RowMajor, 
      int8_t, 
      cutlass::layout::ColumnMajor,
      ElementOutput, 
      cutlass::layout::RowMajor, 
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp, 
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, 
          ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
      3>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = A.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(A.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(B.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(out.data_ptr<int32_t>(), LayoutOutput::packed(output_size));

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha},      1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}

// used by out_proj and fc2, return INT32
torch::Tensor v2_linear_a8_w8_b32_o32CUDA(const torch::Tensor &A,  // INT8
                                   const torch::Tensor &B, // INT8
                                   const torch::Tensor &bias    // INT32
) {
  torch::checkAllSameGPU("linear_a8_w8_b32_o32", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = int32_t;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>; // cutlass::gemm::GemmShape<256, 128, 64>,
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;          // cutlass::gemm::GemmShape<64, 64, 64>, 
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;     // cutlass::gemm::GemmShape<16, 8, 32>,

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, 
      cutlass::layout::RowMajor, 
      int8_t, 
      cutlass::layout::ColumnMajor,
      ElementOutput, 
      cutlass::layout::RowMajor, 
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp, 
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, 
          ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = A.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(A.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(B.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(out.data_ptr<int32_t>(), LayoutOutput::packed(output_size));

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha},      1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}


// used by out_proj and fc2, return INT32
torch::Tensor v1_linear_a8_w8_b32_o32CUDA(const torch::Tensor &A,  // INT8
                                   const torch::Tensor &B, // INT8
                                   const torch::Tensor &bias    // INT32
) {
  torch::checkAllSameGPU("linear_a8_w8_b32_o32", {{A, "A", 0}, {B, "B", 1}, {bias, "bias", 2}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = int32_t;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>; // cutlass::gemm::GemmShape<256, 128, 64>,
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;          // cutlass::gemm::GemmShape<64, 64, 64>, 
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;     // cutlass::gemm::GemmShape<16, 8, 32>,

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, 
      cutlass::layout::RowMajor, 
      int8_t, 
      cutlass::layout::ColumnMajor,
      ElementOutput, 
      cutlass::layout::RowMajor, 
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp, 
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, 
          ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = A.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(A.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(B.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(out.data_ptr<int32_t>(), LayoutOutput::packed(output_size));

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha},      1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}





















torch::Tensor int4MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("int4Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t,                // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      cutlass::int4b_t,                // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int32_t,                         // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {(cutlass::int4b_t *)A.data_ptr<uint8_t>(), K},
      {(cutlass::int4b_t *)B.data_ptr<uint8_t>(), K},
      {C.data_ptr<int32_t>(), N},
      {C.data_ptr<int32_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

torch::Tensor int4OutputInt8MatmulCUDA(const torch::Tensor &A,
                                       const torch::Tensor &B) {
  torch::checkAllSameGPU("int4OutputInt8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt8).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t,                // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      cutlass::int4b_t,                // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int8_t,                          // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {(cutlass::int4b_t *)A.data_ptr<uint8_t>(), K},
      {(cutlass::int4b_t *)B.data_ptr<uint8_t>(), K},
      {C.data_ptr<int8_t>(), N},
      {C.data_ptr<int8_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int32_t,                         // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {C.data_ptr<int32_t>(), N},
      {C.data_ptr<int32_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

torch::Tensor int8OutputInt8MatmulCUDA(const torch::Tensor &A,
                                       const torch::Tensor &B) {
  torch::checkAllSameGPU("int8OutputInt8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt8).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int8_t,                          // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {C.data_ptr<int8_t>(), N},
      {C.data_ptr<int8_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

namespace {
template <typename Gemm>
struct sparseMatmul {
  using ElementComputing = typename Gemm::ElementA;
  using Storage =
      std::conditional_t <
      cutlass::sizeof_bits<ElementComputing>::value<8, uint8_t,
                                                    ElementComputing>;
  static_assert(
      std::is_same<typename Gemm::ElementA, typename Gemm::ElementB>::value);

  static constexpr int kSparse = Gemm::kSparse;
  static constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  using ElementInputE = typename Gemm::ElementE;
  using ElementInputESigned = typename std::make_signed<ElementInputE>::type;
  static_assert(std::is_same<ElementInputE, uint32_t>::value);
  using ReorderedLayoutInputE = typename Gemm::LayoutE;

  static torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B,
                              const torch::Tensor &E) {
    torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
    using cutlassIndex = cutlass::MatrixCoord::Index;
    using pytorchIndex = torch::IntArrayRef::value_type;

    auto M = static_cast<cutlassIndex>(A.size(0));
    auto N = static_cast<cutlassIndex>(B.size(0));
    auto K = static_cast<cutlassIndex>(
        B.size(1) * cutlass::sizeof_bits<Int4Storage>::value /
        cutlass::sizeof_bits<ElementComputing>::value);

    auto C = torch::empty(
        {M, N},
        torch::dtype(util::TorchDtypeDispatcher<typename Gemm::ElementC>::value)
            .device(A.device()));
    const auto extent =
        cutlass::make_Coord(M, K / kSparse / kElementsPerElementE);
    typename Gemm::Arguments arguments{
        {M, N, K},
        {(ElementComputing *)A.data_ptr<Storage>(),
         K / kSparse},                                        // A, lda (sparse)
        {(ElementComputing *)B.data_ptr<Storage>(), K},       // B, ldb
        {C.template data_ptr<typename Gemm::ElementC>(), N},  // C, ldc
        {C.template data_ptr<typename Gemm::ElementC>(), N},  // D, ldd
        {(ElementInputE *)E.data_ptr<ElementInputESigned>(),
         ReorderedLayoutInputE::packed(extent)},  // E, lde (sparse metadata)
        {1, 0},                                   // alpha, beta
        1                                         // split_k_slices
    };

    auto workspaceSize = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    auto workspace =
        torch::empty(static_cast<pytorchIndex>(workspaceSize),
                     torch::dtype(torch::kUInt8).device(A.device()));

    Gemm gemmOp;
    cutlass::Status status;
    status = Gemm::can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                cutlassGetStatusString(status))

    status = gemmOp.initialize(arguments, workspace.data_ptr<uint8_t>());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                cutlassGetStatusString(status))

    status = gemmOp();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                cutlassGetStatusString(status))

    return C;
  }

  static torch::Tensor reorderMeta(const torch::Tensor &E) {
    torch::checkDeviceType("reorderMeta", {E}, torch::DeviceType::CPU);
    using cutlassIndex = cutlass::MatrixCoord::Index;
    auto M = static_cast<cutlassIndex>(E.size(0));
    auto K = static_cast<cutlassIndex>(E.size(1));
    const auto extent = cutlass::make_Coord(M, K);
    auto capacity = ReorderedLayoutInputE::packed(extent).capacity(extent);
    auto out = torch::empty(
        capacity,
        torch::dtype(util::TorchDtypeDispatcher<ElementInputESigned>::value));
    cutlass::TensorRef<ElementInputE, cutlass::layout::RowMajor> tensor_e(
        (ElementInputE *)E.data_ptr<ElementInputESigned>(), K);
    cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
        (ElementInputE *)out.template data_ptr<ElementInputESigned>(),
        ReorderedLayoutInputE::packed(extent));
    cutlass::reorder_meta(tensor_e_reordered, tensor_e, {M, 0, K});

    return out;
  }

  static torch::Tensor genRandomSparseMeta(const int M, const int K) {
    constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
    auto out = torch::empty({M, K / kSparse / kElementsPerElementE},
                            torch::dtype(torch::kInt32));
    const auto extent =
        cutlass::make_Coord(M, K / kSparse / kElementsPerElementE);

    cutlass::TensorView<ElementInputE, cutlass::layout::RowMajor> tensor_e_view{
        (ElementInputE *)out.data_ptr(),
        cutlass::layout::RowMajor::packed(extent), extent};
    cutlass::reference::host::TensorFillRandomSparseMeta(
        tensor_e_view, 1,
        kMetaSizeInBits);  // <- Fill matrix E on host with uniform-distribution
                           // random meta data
    return out;
  }

  static torch::Tensor uncompress(const torch::Tensor &A,
                                  const torch::Tensor &E, const int M,
                                  const int K) {
    auto out =
        torch::empty({M, K * cutlass::sizeof_bits<ElementComputing>::value /
                             cutlass::sizeof_bits<Storage>::value},
                     torch::dtype(util::TorchDtypeDispatcher<Storage>::value));
    cutlass::TensorRef<ElementComputing, cutlass::layout::RowMajor>
        uncompressed_tensor_a{(ElementComputing *)out.data_ptr(), K};
    cutlass::TensorRef<ElementComputing, cutlass::layout::RowMajor> tensor_a{
        (ElementComputing *)A.data_ptr(), K / kSparse};
    cutlass::TensorRef<ElementInputE, cutlass::layout::RowMajor> tensor_e{
        (ElementInputE *)E.data_ptr(), K / kSparse / kElementsPerElementE};
    cutlass::uncompress(uncompressed_tensor_a, tensor_a, tensor_e, M, K);
    return out;
  }
};
using Int4Gemm = cutlass::gemm::device::SparseGemm<
    cutlass::int4b_t,                         // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    cutlass::int4b_t,                         // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    int32_t,                                  // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 256>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 256>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 128>      // ShapeMMAOp
    >;

using Int8Gemm = cutlass::gemm::device::SparseGemm<
    int8_t,                                   // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    int8_t,                                   // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    int32_t,                                  // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 128>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 128>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 64>       // ShapeMMAOp
    >;

using Int4GemmOutputInt8 = cutlass::gemm::device::SparseGemm<
    cutlass::int4b_t,                         // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    cutlass::int4b_t,                         // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    int8_t,                                   // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 256>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 256>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 128>      // ShapeMMAOp
    >;

using Int8GemmOutputInt8 = cutlass::gemm::device::SparseGemm<
    int8_t,                                   // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    int8_t,                                   // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    int8_t,                                   // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 128>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 128>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 64>       // ShapeMMAOp
    >;

template struct sparseMatmul<Int4Gemm>;
template struct sparseMatmul<Int8Gemm>;
template struct sparseMatmul<Int4GemmOutputInt8>;
template struct sparseMatmul<Int8GemmOutputInt8>;
}  // namespace

torch::Tensor int4SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E) {
  return sparseMatmul<Int4Gemm>::matmul(A, B, E);
}

torch::Tensor int4OutputInt8SpMatmulCUDA(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &E) {
  return sparseMatmul<Int4GemmOutputInt8>::matmul(A, B, E);
}

torch::Tensor int4ReorderMeta(const torch::Tensor &E) {
  return sparseMatmul<Int4Gemm>::reorderMeta(E);
}

torch::Tensor int4GenRandomSparseMeta(const int M, const int K) {
  return sparseMatmul<Int4Gemm>::genRandomSparseMeta(M, K);
}

torch::Tensor int4Uncompress(const torch::Tensor &A, const torch::Tensor &E,
                             int M, int K) {
  return sparseMatmul<Int4Gemm>::uncompress(A, E, M, K);
}

torch::Tensor int8SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E) {
  return sparseMatmul<Int8Gemm>::matmul(A, B, E);
}

torch::Tensor int8OutputInt8SpMatmulCUDA(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &E) {
  return sparseMatmul<Int8GemmOutputInt8>::matmul(A, B, E);
}

torch::Tensor int8ReorderMeta(const torch::Tensor &E) {
  return sparseMatmul<Int8Gemm>::reorderMeta(E);
}

torch::Tensor int8GenRandomSparseMeta(const int M, const int K) {
  return sparseMatmul<Int8Gemm>::genRandomSparseMeta(M, K);
}

torch::Tensor int8Uncompress(const torch::Tensor &A, const torch::Tensor &E,
                             int M, int K) {
  return sparseMatmul<Int8Gemm>::uncompress(A, E, M, K);
}

#ifdef QUIK_WITH_CUSPARSELT
#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
    }                                                                          \
  }

CusparseLtInt8SpMatmul::CusparseLtInt8SpMatmul(const torch::Tensor &A,
                                               const torch::Tensor &B,
                                               const int alg = 0)
    : A_(A),
      B_(B),
      handle_(new cusparseLtHandle_t),
      matA_(new cusparseLtMatDescriptor_t),
      matB_(new cusparseLtMatDescriptor_t),
      matC_(new cusparseLtMatDescriptor_t),
      matmul_(new cusparseLtMatmulDescriptor_t),
      alg_sel_(new cusparseLtMatmulAlgSelection_t),
      plan_(new cusparseLtMatmulPlan_t) {
  torch::checkAllContiguous("CusparseLtInt8SpMatmul",
                            {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("CusparseLtInt8SpMatmul", {A, B},
                         at::DeviceType::CUDA);
  torch::checkAllSameGPU("CusparseLtInt8SpMatmul", {{A, "A", 0}, {B, "B", 1}});
  M_ = A_.size(0);
  K_ = A_.size(1);
  N_ = B_.size(0);
  dA_ = A_.data_ptr<int8_t>();
  dB_ = B_.data_ptr<int8_t>();

  alpha_ = 1.0f;
  beta_ = 0.0f;

  cudaDataType_t input_type, output_type;
  cusparseComputeType compute_type;

  cusparseOrder_t order;
  bool is_rowmajor;
  bool isA_transposed, isB_transposed;
  int64_t num_A_rows, num_A_cols;
  int64_t num_B_rows, num_B_cols;
  int64_t num_C_rows, num_C_cols;
  unsigned alignment;
  int64_t lda, ldb, ldc;

  order = CUSPARSE_ORDER_ROW;
  opA_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  opB_ = CUSPARSE_OPERATION_TRANSPOSE;
  input_type = CUDA_R_8I;
  output_type = CUDA_R_16F;
  compute_type = CUSPARSE_COMPUTE_32I;

  is_rowmajor = (order == CUSPARSE_ORDER_ROW);
  isA_transposed = (opA_ != CUSPARSE_OPERATION_NON_TRANSPOSE);
  isB_transposed = (opB_ != CUSPARSE_OPERATION_NON_TRANSPOSE);
  num_A_rows = (isA_transposed) ? K_ : M_;
  num_A_cols = (isA_transposed) ? M_ : K_;
  num_B_rows = (isB_transposed) ? N_ : K_;
  num_B_cols = (isB_transposed) ? K_ : N_;
  num_C_rows = M_;
  num_C_cols = N_;
  alignment = 16;
  lda = (is_rowmajor) ? num_A_cols : num_A_rows;
  ldb = (is_rowmajor) ? num_B_cols : num_B_rows;
  ldc = (is_rowmajor) ? num_C_cols : num_C_rows;

  num_streams_ = 0;

  CHECK_CUSPARSE(cusparseLtInit(handle_))
  // matrix descriptor initialization
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      handle_, matA_, num_A_rows, num_A_cols, lda, alignment, input_type, order,
      CUSPARSELT_SPARSITY_50_PERCENT))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(handle_, matB_, num_B_rows,
                                               num_B_cols, ldb, alignment,
                                               input_type, order))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(handle_, matC_, num_C_rows,
                                               num_C_cols, ldc, alignment,
                                               output_type, order))
  // matmul, algorithm selection, and plan initialization

  CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
      handle_, matmul_, opA_, opB_, matA_, matB_, matC_, matC_, compute_type))

  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
      handle_, alg_sel_, matmul_, CUSPARSELT_MATMUL_ALG_DEFAULT))

  CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
      handle_, alg_sel_, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(handle_, plan_, matmul_, alg_sel_))
}

CusparseLtInt8SpMatmul::~CusparseLtInt8SpMatmul() {
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(matA_))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(matB_))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(matC_))
  CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(plan_))
  CHECK_CUSPARSE(cusparseLtDestroy(handle_))
}

void CusparseLtInt8SpMatmul::compress() {
  size_t compressed_size, compressed_buffer_size;
  CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(
      handle_, matA_, &compressed_size, &compressed_buffer_size))

  A_compressed_ = torch::empty(static_cast<pytorchIndex>(compressed_size),
                               torch::dtype(torch::kInt8).device(torch::kCUDA));
  dA_compressed_ = A_compressed_.data_ptr<int8_t>();

  auto A_compressedBuffer =
      torch::empty(static_cast<pytorchIndex>(compressed_buffer_size),
                   torch::dtype(torch::kUInt8).device(torch::kCUDA));
  void *dA_compressedBuffer = A_compressedBuffer.data_ptr<uint8_t>();

  CHECK_CUSPARSE(cusparseLtSpMMACompress2(handle_, matA_, true, opA_, dA_,
                                          dA_compressed_, dA_compressedBuffer,
                                          stream_))
}

torch::Tensor CusparseLtInt8SpMatmul::matmulDefault() {
  auto C = torch::empty({M_, N_},
                        torch::dtype(torch::kFloat16).device(torch::kCUDA));
  half *dC = (half *)C.data_ptr<torch::Half>();
  half *dD = dC;

  size_t workspace_size;

  CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(handle_, plan_, &workspace_size))

  auto workspace =
      torch::empty(static_cast<pytorchIndex>(workspace_size),
                   torch::dtype(torch::kUInt8).device(torch::kCUDA));
  void *d_workspace = workspace.data_ptr<uint8_t>();

  cusparseLtMatmul(handle_, plan_, &alpha_, dA_compressed_, dB_, &beta_, dC, dD,
                   d_workspace, streams_, num_streams_);
  return C;
}

torch::Tensor CusparseLtInt8SpMatmul::matmulBy(const torch::Tensor &B) {
  TORCH_CHECK(B.device() == A_compressed_.device())
  TORCH_CHECK(B.size(1) == K_)
  int8_t *dB = B.data_ptr<int8_t>();

  auto C = torch::empty({M_, N_},
                        torch::dtype(torch::kFloat16).device(torch::kCUDA));
  half *dC = (half *)C.data_ptr<torch::Half>();
  half *dD = dC;

  size_t workspace_size;

  CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(handle_, plan_, &workspace_size))

  auto workspace =
      torch::empty(static_cast<pytorchIndex>(workspace_size),
                   torch::dtype(torch::kUInt8).device(torch::kCUDA));
  void *d_workspace = workspace.data_ptr<uint8_t>();

  cusparseLtMatmul(handle_, plan_, &alpha_, dA_compressed_, dB, &beta_, dC, dD,
                   d_workspace, streams_, num_streams_);
  return C;
}

#endif
}  // namespace QUIK::matmul