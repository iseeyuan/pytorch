#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <torch/csrc/autograd/symbolic.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/onnx/onnx.h>

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/pickler.h>
#include <torch/csrc/jit/generic_instruction.h>

#include <caffe2/core/types.h>
//#include <caffe2/proto/caffe2_pb.h>
//#include <caffe2/proto/torch_pb.h>
#include <caffe2/proto/instruction_pb.h>
#include <caffe2/serialize/inline_container.h>
#include <onnx/onnx_pb.h>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

namespace torch {
namespace jit {

// Shared with ScriptModuleSerializer
void convertAndWriteTensor(
    size_t tensor_id,
    const at::Tensor& tensor,
    torch::TensorDef* tensor_proto,
    std::unordered_map<const void*, std::string>& storageMap,
    caffe2::serialize::PyTorchStreamWriter& writer);

namespace {

// this is a serializer class which saves instructions and constants to a file.
// It's similar to file structure in InstructionsSerializer.
// TODO: modularize the attribute and tensor parts in the final version, if the's
// no difference on tensors in models and instructions.
class InstructionsSerializer final {
 public:
  InstructionsSerializer(const std::string& filename);

  InstructionsSerializer(std::ostream* ofs);

  void serialize(const GenericInstructionList& inslist);

 private:
  std::ofstream ofs_;
  caffe2::serialize::PyTorchStreamWriter writer_;

  // all tensors that will be stored
  std::vector<at::Tensor> tensor_table_;
};

// InstructionsSerializer's methods
InstructionsSerializer::InstructionsSerializer(const std::string& filename)
    : writer_(filename.c_str()) {
  // TODO appropriate support for mmap, right now we still use stream writer
}

InstructionsSerializer::InstructionsSerializer(std::ostream* ofs)
    : ofs_(), writer_(ofs) {}

void InstructionsSerializer::serialize(const GenericInstructionList& inslist) {
  instruction::InstructionListProto list_proto;
  list_proto.set_name("list");

  // instructions and tensors
  std::unordered_map<const void*, std::string> storageMap;
  size_t tensor_id = 0;
  for (const auto& ins : inslist.instructions) {
    auto ins_proto = list_proto.add_instructions();

    // operator
    auto op = ins_proto->mutable_op();
    std::cout << "Exporting op " << ins.name << std::endl;
    op->set_name(ins.name);
    op->set_overload_name(ins.overload_name);

    // inputs
    for (const auto& input : ins.inputs) {
      auto input_proto = ins_proto->add_inputs();
      input_proto->set_unique_id(input.unique_id);
      input_proto->set_free_flag(input.free_flag);
    }

    // outputs
    for (const auto& output : ins.outputs) {
      auto output_proto = ins_proto->add_outputs();
      output_proto->set_unique_id(output.unique_id);
    }

    // attributes
    for (const auto& val : ins.attributes) {
      auto attribute = ins_proto->add_attributes();
      if (val.isInt()) {
        attribute->set_kind(instruction::AttributeValueProto::i);
        attribute->set_int_value(val.toInt());
      }
      else if (val.isDouble()) {
        attribute->set_kind(instruction::AttributeValueProto::f);
        attribute->set_float_value(val.toDouble());
      }
      else if (val.isTensor()) {
        attribute->set_kind(instruction::AttributeValueProto::t);
        attribute->set_tensor_id(tensor_id);
        auto tensor_proto = list_proto.add_tensors();
        convertAndWriteTensor(tensor_id++, val.toTensor(), tensor_proto, storageMap, writer_);
      }
      else if (val.isIntList()) {
        attribute->set_kind(instruction::AttributeValueProto::is);
        for (auto element : val.toIntList()->elements()) {
          attribute->add_int_list(element);
        }
      }
      else if (val.isDoubleList()) {
        attribute->set_kind(instruction::AttributeValueProto::fs);
        for (auto element : val.toDoubleList()->elements()) {
          attribute->add_float_list(element);
        }
      }
      else if (val.isBool()) {
        attribute->set_kind(instruction::AttributeValueProto::b);
        attribute->set_bool_value(val.toBool());
      }
      else if (val.isBoolList()) {
        attribute->set_kind(instruction::AttributeValueProto::bs);
        for (auto element : val.toBoolList()->elements()) {
          attribute->add_bool_list(element);
        }
      }
      else if (val.isNone()) {
        attribute->set_kind(instruction::AttributeValueProto::n);
      }
      else {
        throw std::runtime_error("Value type of Constant operator is not supported yet.");
      }
    }
  }

  std::string output;
  // NB: cannot use MessageToJsonString, since fbcode's protobuf is too old
  // be consistent with MessageToJsonString
  std::string url_prefix = "type.googleapis.com";
  std::unique_ptr<::google::protobuf::util::TypeResolver> resolver(
      ::google::protobuf::util::NewTypeResolverForDescriptorPool(
          url_prefix, list_proto.GetDescriptor()->file()->pool()));
  ::google::protobuf::util::Status convert_result =
      ::google::protobuf::util::BinaryToJsonString(
          resolver.get(),
          url_prefix + "/" + list_proto.GetDescriptor()->full_name(),
          list_proto.SerializeAsString(),
          &output);
  if (!convert_result.ok()) {
    std::stringstream ss;
    ss << convert_result;
    AT_ERROR(ss.str());
  }
  std::cout << output << std::endl;
  writer_.writeRecord("instructions.json", output.data(), output.size());
  writer_.writeEndOfFile();
}

}

void ExportInstructions(
    const GenericInstructionList& inslist,
    std::ostream& os) {
  InstructionsSerializer serializer(&os);
  serializer.serialize(inslist);
}

} // namespace jit
} // namespace torch
