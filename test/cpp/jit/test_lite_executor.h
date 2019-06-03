#pragma once

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/jit.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/script.h"

#include <torch/csrc/lite_interpreter/import_instructions.h>
#include <torch/csrc/jit/generic_instruction.h>
#include <torch/csrc/lite_interpreter/instruction_executor.h>

#include <ATen/ATen.h>

namespace torch {
namespace jit {
namespace test {

void testLiteExecutor() {
  auto m = std::make_shared<script::Module>();
  m->register_parameter("foo", torch::ones({}), false);
  m->define(R"(
    def add_it(self, x, b : int = 4):
      return self.foo + x + b
  )");
  m->eval();

  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m->run_method("add_it", minput);

  std::stringstream ss;
  m->save_method("add_it", inputs, ss);
//  m->save_method("add_it", inputs, "/Users/myuan/data/add_it.bc");
  // Load and execute
  std::shared_ptr<torch::jit::GenericInstructionList> list = torch::jit::loadInstructionList(ss);
  torch::jit::InstructionExecutor executor(list);
  auto res = executor.run(inputs);

  AT_ASSERT(res.toTensor().item<float>() == ref.toTensor().item<float>());

  // TODO:
  // 1. Load ss to a InstructionList
  // 2. Execute InstructionList
  // 3. Compare the result to n->run_method("add_it", torch::ones({})).toTensor()
//  AT_ASSERT(!ss.str().empty());
}

} // namespace test
} // namespace jit
} // namespace torch
