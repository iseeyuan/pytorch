#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/stack.h>

using Stack = std::vector<c10::IValue>;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;

static auto registry0 = c10::RegisterOperators().op(
  "aten::matmul",
  c10::kernel<decltype(at::matmul), &at::matmul>(),
  c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::add_Tensor_Tensor_Scalar__Tensor",
  c10::kernel([](at::Tensor a, at::Tensor b, at::Scalar c) ->at::Tensor {
    return at::add(a, b, c);
  }),
  c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::add_Tensor_Scalar_Scalar__Tensor",
  c10::kernel([](at::Tensor a, at::Scalar b, at::Scalar c) ->at::Tensor {
  return at::add(a, b, c);
                                }),
                                c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::add__Tensor_Tensor_Scalar__Tensor",
  c10::kernel([](at::Tensor a, at::Tensor b, at::Scalar c) ->at::Tensor {
  return at::add(a, b, c);
  }),
  c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::adaptive_avg_pool2d_Tensor_int[]__Tensor",
  c10::kernel([](at::Tensor a, c10::IntArrayRef b) ->at::Tensor {
  return at::adaptive_avg_pool2d(a, b);
  }),
  c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::mm_Tensor_Tensor__Tensor",
  c10::kernel([](at::Tensor a, at::Tensor b) ->at::Tensor {
  return at::mm(a, b);
                                }),
                                c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::_convolution_Tensor_Tensor_Tensor?_int[]_int[]_int[]_bool_int[]_int_bool_bool_bool__Tensor",
  c10::kernel<decltype(at::_convolution), &at::_convolution>(),
                                c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::batch_norm_Tensor_Tensor?_Tensor?_Tensor?_Tensor?_bool_float_float_bool__Tensor",
  c10::kernel<decltype(at::batch_norm), &at::batch_norm>(),
                                c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::max_pool2d_with_indices_Tensor_int[]_int[]_int[]_int[]_bool__Tensor_Tensor",
  c10::kernel<decltype(at::max_pool2d_with_indices), &at::max_pool2d_with_indices>(),
                                c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::relu_Tensor__Tensor",
  c10::kernel<decltype(at::relu), &at::relu>(),
                                c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::t_Tensor__Tensor",
  c10::kernel<decltype(at::t), &at::t>(),
                                c10::dispatchKey(at::CPUTensorId())
).op(
  "aten::size_Tensor_int__int",
  c10::kernel<decltype(at::size), &at::size>(),
                                c10::dispatchKey(at::CPUTensorId())
).op(
    "prim::Load___",
    c10::kernel([]() {
    })
).op(
    "prim::Store___",
    c10::kernel([]() {
    })
);


//class MyKernel : public OperatorKernel {
// public:
//  MyKernel(int value): value_(value) {}
//  int operator()() {
//    return value_;
//  }
//};

//static auto registry2 = c10::RegisterOperators().op(
//    "aten::constant6",
//    c10::kernel<MyKernel>(6)
//);
