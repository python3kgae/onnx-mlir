//====------ ConvertKrnlToSPIRV.cpp - Krnl Dialect Lowering  ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of Krnl operations to a combination of
// other dialects (affine, std, LLVM).
//
//===----------------------------------------------------------------------===//


#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

#include <memory>

using namespace mlir;

#define DEBUG_TYPE "krnl_to_spirv"

namespace onnx_mlir {
namespace krnl {

//===----------------------------------------------------------------------===//
// Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

struct ConvertKrnlToSPIRVPass
    : public PassWrapper<ConvertKrnlToSPIRVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertKrnlToSPIRVPass)

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ConvertKrnlToSPIRVPass() = default;
  ConvertKrnlToSPIRVPass(const ConvertKrnlToSPIRVPass &pass)
      : PassWrapper<ConvertKrnlToSPIRVPass, OperationPass<ModuleOp>>() {}
  ConvertKrnlToSPIRVPass(bool verifyInputTensors, bool useOpaquePointers,
      bool useLRODATA, bool storeConstantsToFile,
      uint64_t constantsToFileSingleThreshold,
      uint64_t constantsToFileTotalThreshold, std::string outputNameNoExt,
      bool enableParallel) {
    this->verifyInputTensors = verifyInputTensors;
    this->useOpaquePointers = useOpaquePointers;
    // Exclusive options. no option or only one option can be True.
    this->useLRODATA = useLRODATA;
    this->storeConstantsToFile = storeConstantsToFile;
    this->constantsToFileSingleThreshold = constantsToFileSingleThreshold;
    this->constantsToFileTotalThreshold = constantsToFileTotalThreshold;
    this->outputNameNoExt = outputNameNoExt;
    this->enableParallel = enableParallel;
  }

  StringRef getArgument() const override { return "convert-krnl-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the Krnl Affine and Std dialects to LLVM.";
  }

  void runOnOperation() final;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect>();
  }

  Option<bool> useOpaquePointers{*this, "use-opaque-pointers",
      llvm::cl::desc("Whether to use opaque pointers instead of typed pointers "
                     "when lowering to LLVM. Default: true"),
      llvm::cl::init(true)};

  Option<bool> verifyInputTensors{*this, "verify-input-tensors",
      llvm::cl::desc(
          "Verify input tensors whenever the entry point function is called.\n"
          "Data type and shape are verified. Enable this may introduce "
          "overhead in inferencing."),
      llvm::cl::init(false)};

  Option<bool> useLRODATA{*this, "use-lrodata-section",
      llvm::cl::desc("Put global constants into the large read-only data "
                     "section. This is for linking large object files"),
      llvm::cl::init(false)};

  Option<bool> storeConstantsToFile{*this, "store-constants-to-file",
      llvm::cl::desc("Put global constants to a file."), llvm::cl::init(false)};

  Option<float> constantsToFileTotalThreshold{*this,
      "constants-to-file-total-threshold",
      llvm::cl::desc(
          "Put global constants to a file if the total size in "
          "bytes of constants is greater than this threshold. "
          "store-constants-to-file must be enabled for this to be effective. "
          "Only count contants whose size is greater than "
          "constants-to-file-single-threshold. Value is in GB."),
      llvm::cl::init(2.0)};

  Option<float> constantsToFileSingleThreshold{*this,
      "constants-to-file-single-threshold",
      llvm::cl::desc(
          "Put global constants to a file if a single constant's size in "
          "bytes is greater than this threshold. "
          "store-constants-to-file must be enabled for this to be effective. "
          "Total sizes in bytes of satisfied constants must be greater than "
          "constants-to-file-total-threshold. Value is in KB."),
      llvm::cl::init(1.0)};

private:
  std::string outputNameNoExt = "./model";
  bool enableParallel;
};

void ConvertKrnlToSPIRVPass::runOnOperation() {
//   ModuleOp module = getOperation();
//   MLIRContext *ctx = &getContext();
//   OpBuilder builder(ctx);
//   const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
//   LowerToLLVMOptions options(ctx, dataLayoutAnalysis.getAtOrAbove(module));

//   // MLIR/LLVM is moving to using opaque pointers instead of typed pointers.
//   // Remove this once MLIR/LLVM completely uses opaque pointers.
//   options.useOpaquePointers = useOpaquePointers; // for LLVMTypeConverter.
//   LLVM_USE_OPAQUE_POINTER = useOpaquePointers; // for onnx-mlir util functions.

//   // Append a unique string to each entry point function.
//   // The string is getting from the module's attribute
//   // `onnx-mlir.symbol-postfix`.
//   PostfixEntrypointNames(module);

//   KRNL_ENTRY_POINT_ID = 0;

//   // Global Op for entry point names and their input/output JSON signatures,
//   // those will generated when lowering KrnlEntryPoint.
//   // This info is used to generate global signature functions.
//   SmallVector<LLVM::GlobalOp, 1> entryGlobalOps, inSigGlobalOps,
//       outSigGlobalOps;

//   // Keep original MemRefTypes for inputs and outputs. These information will be
//   // used for constructing OMTensors for inputs and outputs.
//   // We have to record this information at this point before they are
//   // disappeared during the lowering to LLVM. For example, unsigned types do
//   // not exist at LLVM level, typed pointers becomes opaque if opaque point is
//   // enabled.
//   std::map<std::string, SmallVector<MemRefType, 4>> inputMemRefTypes;
//   std::map<std::string, SmallVector<MemRefType, 4>> outputMemRefTypes;
//   recordInputOutputMemRefTypes(module, inputMemRefTypes, outputMemRefTypes);

//   // Determine whether the module has a single entry point or not.
//   bool singleEntryPoint = hasSingleEntryPoint(module);

//   // Determine whether an output OMTensor should own the underlying buffer or
//   // not.
//   SmallVector<bool, 4> outputOMTensorOwnerships;
//   determineOwnershipForOutputOMTensors(module, outputOMTensorOwnerships);

//   // If storeConstantsToFile, copy constants from GlobalOp and write to a single
//   // file.
//   // A single constant's size must be greater than singleThreshold.
//   // The total size of contants must be greater than totalThreshold.
//   std::string fname = outputNameNoExt + ".constants.bin";
//   if (storeConstantsToFile) {
//     storeConstantsToFile = extractConstantsToFile(module, fname,
//         (uint64_t)constantsToFileSingleThreshold * 1024,
//         (uint64_t)constantsToFileTotalThreshold * 1024 * 1024 * 1024);
//   }

//   // Request C wrapper emission via attribute.
//   for (auto func : module.getOps<func::FuncOp>()) {
//     func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
//         UnitAttr::get(&getContext()));
//   }

//   // Define the target for this lowering i.e. the LLVM dialect.
//   ConversionTarget target(*ctx);
//   target.addLegalDialect<LLVM::LLVMDialect>();
//   target.addLegalOp<ModuleOp>();
//   target.addLegalOp<UnrealizedConversionCastOp>();

//   // Conversion target for accelerators.
//   for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
//     accel->conversionTargetKrnlToLLVM(target);

//   // Convert types to legal types for the LLVM dialect.
//   LLVMTypeConverter typeConverter(ctx, options);
//   customizeTypeConverter(typeConverter);

//   // omp::ParallelOp can only be legalized when its region is legal
//   target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
//       [&](Operation *op) { return typeConverter.isLegal(&op->getRegion(0)); });
//   // Currently, only minimum required OpenMP Ops are marked as legal, in the
//   // future integration of OpenMP, probably more OpenMP Ops are required to be
//   // marked as legal. Please refer the Conversion/OpenMPToLLVM/OpenMPtoLLVM.cpp
//   // in MLIR repo to see see how to legalize them.
//   target.addLegalOp<omp::TerminatorOp, omp::YieldOp>();
//   // We have a combination of `krnl`, `affine`, `vector`, and `std` operations.
//   // We lower in stages until all the code is in the LLVM dialect.
//   RewritePatternSet patterns(ctx);

//   populateAffineAndKrnlToLLVMConversion(patterns, typeConverter, ctx,
//       outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
//       inSigGlobalOps, outSigGlobalOps, inputMemRefTypes, outputMemRefTypes,
//       verifyInputTensors, enableParallel);

//   // Rewrite patterns for accelerators.
//   for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
//     accel->rewritePatternKrnlToLLVM(patterns, typeConverter, ctx);

//   // We want to completely lower to LLVM, so we use a `FullConversion`. This
//   // ensures that only legal operations will remain after the conversion.
//   if (failed(
//           applyFullConversion(getOperation(), target, std::move(patterns)))) {
//     signalPassFailure();
//   }

//   // Generate signature functions.
//   if (entryGlobalOps.size() >= 1)
//     genSignatureFunction(
//         module, entryGlobalOps, inSigGlobalOps, outSigGlobalOps);

//   // If globals are stored on external files. Emit helper functions to load
//   // constants from files.
//   if (storeConstantsToFile) {
//     // Register runtime function calls, e.g. omXXX functions.
//     const RuntimeAPIRegistry &apiRegistry =
//         RuntimeAPIRegistry(module, builder, typeConverter);

//     // Emit a function, omLoadConstantsFromFile, that loads contants from files
//     // to memory.
//     loadConstantsFromFile(module, apiRegistry, entryGlobalOps);
//   }

//   // Annotate global constants with `.lrodata` section if required.
//   // Make sure this is always called at the end of this pass.
//   if (useLRODATA) {
    // module->walk([&](LLVM::GlobalOp gop) -> WalkResult {
    //   // Put all global constants into `.lrodata` instead of `.rodata` because
    //   // AI workloads often have a large amount of constants, especially large
    //   // language models.
    //   gop.getOperation()->setAttr("section", StringAttr::get(ctx, ".lrodata"));
    //   return WalkResult::advance();
    // });
//   }
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to SPIRV.
std::unique_ptr<Pass> createConvertKrnlToSPIRVPass() {
  return std::make_unique<ConvertKrnlToSPIRVPass>();
}
std::unique_ptr<Pass> createConvertKrnlToSPIRVPass(bool verifyInputTensors,
    bool useOpaquePointers, bool useLRODATA, bool storeConstantsToFile,
    float constantsToFileSingleThreshold, float constantsToFileTotalThreshold,
    std::string outputNameNoExt, bool enableParallel) {
  return std::make_unique<ConvertKrnlToSPIRVPass>(verifyInputTensors,
      useOpaquePointers, useLRODATA, storeConstantsToFile,
      constantsToFileSingleThreshold, constantsToFileTotalThreshold,
      outputNameNoExt, enableParallel);
}

} // namespace krnl
} // namespace onnx_mlir
