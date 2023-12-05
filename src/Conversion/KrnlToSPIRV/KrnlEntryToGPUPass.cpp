//===- KrnlEntryToGPUPass.cpp - Func to Krnl entry to GPU Passes ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert Krnl dialect entry to GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "KrnlEntryToGPU.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "src/Pass/Passes.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;
using namespace mlir::gpu;

namespace {
/// A pass converting MLIR Krnl entrys into the GPU dialect.
class ConvertKrnlEntryToGPUPass : public OperationPass<mlir::ModuleOp> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertKrnlEntryToGPUPass)

  ConvertKrnlEntryToGPUPass()
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<ConvertKrnlEntryToGPUPass>()) {}
  ConvertKrnlEntryToGPUPass(const ConvertKrnlEntryToGPUPass &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("convert-gpu-launch-to-vulkan-launch");
  }
  ::llvm::StringRef getArgument() const override {
    return "convert-gpu-launch-to-vulkan-launch";
  }

  ::llvm::StringRef getDescription() const override {
    return "Convert gpu.launch_func to vulkanLaunch external call";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ConvertGpuLaunchFuncToVulkanLaunchFunc");
  }
  ::llvm::StringRef getName() const override {
    return "ConvertGpuLaunchFuncToVulkanLaunchFunc";
  }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() ==
           ::mlir::TypeID::get<ConvertKrnlEntryToGPUPass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertKrnlEntryToGPUPass>(
        *static_cast<const ConvertKrnlEntryToGPUPass *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {

    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void ConvertKrnlEntryToGPUPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  KrnlEntryPointOp entryPointOp;
  auto walkResult = module->walk([&](mlir::Operation *op) -> WalkResult {
    if (auto entryOp = llvm::dyn_cast<KrnlEntryPointOp>(op)) {
      entryPointOp = entryOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Do nothing if there is no EntryPoint.
  if (!walkResult.wasInterrupted())
    return;

  // Get the entry point function.
  std::string entryPointFuncName =
      entryPointOp
          ->getAttrOfType<SymbolRefAttr>(
              KrnlEntryPointOp::getEntryPointFuncAttrName())
          .getLeafReference()
          .getValue()
          .str();
  func::FuncOp entryPointFunc =
      dyn_cast<func::FuncOp>(module.lookupSymbol(entryPointFuncName));
  assert(entryPointFunc && "entry point func must exist");

  auto numInputs =
      entryPointOp
          ->getAttrOfType<IntegerAttr>(KrnlEntryPointOp::getNumInputsAttrName())
          .getInt();
  if (numInputs) {
  
  }
  auto numOutputs = entryPointOp
                        ->getAttrOfType<IntegerAttr>(
                            KrnlEntryPointOp::getNumOutputsAttrName())
                        .getInt();
  if (numOutputs) {
  
  }
  // Convert Krnl entry point to GPU entry point.
  //auto gpuModule = convertKrnlEntryToGPUModule(module, EntryOp);

  TypeConverter typeConverter = TypeConverter();
  RewritePatternSet patterns(context);
  populateKrnlEntryToGPUPatterns(typeConverter, patterns);
  // populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);
  ConversionTarget target(getContext());

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
onnx_mlir::krnl::createConvertKrnlEntryToGPUPass() {
  return std::make_unique<ConvertKrnlEntryToGPUPass>();
}
