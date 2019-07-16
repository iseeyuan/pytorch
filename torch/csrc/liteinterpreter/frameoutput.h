#pragma once
#include <torch/csrc/jit/instruction.h>
#include <memory>
#include <vector>

//#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/source_range.h>
#include <ATen/core/stack.h>
#include <ATen/core/interned_strings.h>

namespace torch {
namespace jit {

//struct Variable {
//  size_t unique_id;
//  bool free_flag;
//};

//struct OperatorOutput {
//  // For C10 dispatcher and debugging
//  std::string name;
//  std::string overload_name;
//};

// A subset version of ActiveFrame. It does not have Function, which
// depends on graph. It does not have types at this time. It's used for
// serialization/deserialization purpose.
struct FrameOutput {
  std::string name;
  size_t pc;
  std::vector<Instruction> instructions;
  std::vector<IValue> constants;
  std::vector<Operation> operators;
};

} // namespace jit
} // namespace torch
