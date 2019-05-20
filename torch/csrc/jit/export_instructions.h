#pragma once
#include <vector>
#include <torch/csrc/jit/generic_instruction.h>

namespace torch {
namespace jit {

struct Instruction;

void ExportInstructions(
    const GenericInstructionList& inslist,
    std::ostream& os);

} // namespace jit
} // namespace torch
