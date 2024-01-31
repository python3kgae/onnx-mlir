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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"

#include "src/Pass/Passes.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/TypeSwitch.h"



#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"



#include <optional>

#define DEBUG_TYPE "krnl-entry-to-gpu"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::affine;


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
    return ::llvm::StringLiteral("convert-krnl-entry-to-gpu");
  }
  ::llvm::StringRef getArgument() const override {
    return "convert-krnl-entry-to-gpu";
  }

  ::llvm::StringRef getDescription() const override {
    return "Convert KrnlEntry to GPU";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ConvertKrnlEntryToGPUPass");
  }
  ::llvm::StringRef getName() const override {
    return "ConvertKrnlEntryToGPUPass";
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


// affineLoopToGPU convert

// Check the structure of the loop nest:
//   - there are enough loops to map to numDims;
//   - the loops are perfectly nested;
//   - the loop bounds can be computed above the outermost loop.
// This roughly corresponds to the "matcher" part of the pattern-based
// rewriting infrastructure.
static LogicalResult checkAffineLoopNestMappableImpl(
    AffineForOp forOp, unsigned numDims) {
  Region &limit = forOp.getRegion();
  for (unsigned i = 0, e = numDims; i < e; ++i) {
    Operation *nested = &forOp.getBody()->front();
    if (!areValuesDefinedAbove(forOp.getLowerBoundOperands(), limit) ||
        !areValuesDefinedAbove(forOp.getUpperBoundOperands(), limit))
      return forOp.emitError(
          "loops with bounds depending on other mapped loops "
          "are not supported");

    // The innermost loop can have an arbitrary body, skip the perfect nesting
    // check for it.
    if (i == e - 1)
      break;

    auto begin = forOp.getBody()->begin(), end = forOp.getBody()->end();
    if (forOp.getBody()->empty() || std::next(begin, 2) != end)
      return forOp.emitError("expected perfectly nested loops in the body");

    if (!(forOp = dyn_cast<AffineForOp>(nested)))
      return nested->emitError("expected a nested loop");
  }
  return success();
}

static LogicalResult checkAffineLoopNestMappable(
    AffineForOp forOp, unsigned numBlockDims, unsigned numThreadDims) {
  if (numBlockDims < 1 || numThreadDims < 1) {
    LLVM_DEBUG(llvm::dbgs() << "nothing to map");
    return success();
  }

  if (numBlockDims > 3) {
    return forOp.emitError("cannot map to more than 3 block dimensions");
  }
  if (numThreadDims > 3) {
    return forOp.emitError("cannot map to more than 3 thread dimensions");
  }
  return checkAffineLoopNestMappableImpl(forOp, numBlockDims + numThreadDims);
}

SmallVector<AffineForOp, 4> collectAffineLoopNestBounds(AffineForOp forOp) {
  OpBuilder builder(forOp.getOperation());
  AffineForOp currentLoop = forOp;
  SmallVector<AffineForOp, 4> loops;

  while (true) {
    if (!currentLoop.hasConstantBounds())
              break;
    auto lb = currentLoop.getConstantLowerBound();
    auto ub = currentLoop.getConstantUpperBound();
    auto step = currentLoop.getStep();
    lb++;
    ub++;

    if (currentLoop.getBody()->getOperations().size() != 2) {
      loops.push_back(currentLoop);
      break;
    }

    if (!isa<AffineYieldOp>(&currentLoop.getBody()->back()))
      break;

    if (auto nestForOp =
            dyn_cast<AffineForOp>(&currentLoop.getBody()->front())) {
      loops.push_back(currentLoop);
      currentLoop = nestForOp;
    } else {
      break;
    }
  }
  return loops;
}

namespace {
// Helper structure that holds common state of the loop to GPU kernel
// conversion.
struct AffineLoopToGpuConverter {
  std::optional<affine::AffineForOp> collectBounds(
      affine::AffineForOp forOp, unsigned numLoops);

  void createGlobalID(affine::AffineForOp rootForOp,
      affine::AffineForOp innermostForOp,
      unsigned numBlockDims, unsigned numThreadDims);

  // Ranges of the loops mapped to blocks or threads.
  SmallVector<Value, 6> dims;
  // Lower bounds of the loops mapped to blocks or threads.
  SmallVector<Value, 6> lbs;
  // Induction variables of the loops mapped to blocks or threads.
  SmallVector<Value, 6> ivs;
  // Steps of the loops mapped to blocks or threads.
  SmallVector<Value, 6> steps;
  unsigned NumGroups = 1;
  unsigned GroupSize = 1;
};
} // namespace

// Extract an indexed value from KernelDim3.
static Value getDim3Value(const gpu::KernelDim3 &dim3, unsigned pos) {
  switch (pos) {
  case 0:
    return dim3.x;
  case 1:
    return dim3.y;
  case 2:
    return dim3.z;
  default:
    llvm_unreachable("dim3 position out of bounds");
  }
  return nullptr;
}

// Collect ranges, bounds, steps and induction variables in preparation for
// mapping a loop nest of depth "numLoops" rooted at "forOp" to a GPU kernel.
// This may fail if the IR for computing loop bounds cannot be constructed, for
// example if an affine loop uses semi-affine maps. Return the last loop to be
// mapped on success, std::nullopt on failure.
std::optional<AffineForOp> AffineLoopToGpuConverter::collectBounds(
    AffineForOp forOp, unsigned numLoops) {
  OpBuilder builder(forOp.getOperation());
  dims.reserve(numLoops);
  lbs.reserve(numLoops);
  ivs.reserve(numLoops);
  steps.reserve(numLoops);
  AffineForOp currentLoop = forOp;
  for (unsigned i = 0; i < numLoops; ++i) {
    Value lowerBound = lowerAffineLowerBound(currentLoop, builder);
    Value upperBound = lowerAffineUpperBound(currentLoop, builder);
    if (!lowerBound || !upperBound) {
      return std::nullopt;
    }

    Value range = builder.create<arith::SubIOp>(
        currentLoop.getLoc(), upperBound, lowerBound);

    Value step = builder.create<arith::ConstantIndexOp>(
        forOp.getLoc(), forOp.getStepAsInt());

    if (getConstantIntValue(step) != static_cast<int64_t>(1))
      range = builder.create<arith::DivSIOp>(currentLoop.getLoc(), range, step);
    dims.push_back(range);

    lbs.push_back(lowerBound);
    ivs.push_back(currentLoop.getInductionVar());
    steps.push_back(step);

    if (i != numLoops - 1)
      currentLoop = cast<AffineForOp>(&currentLoop.getBody()->front());
  }
  return currentLoop;
}

// Replace the rooted at "rootForOp" with a GPU launch operation.  This expects
// "innermostForOp" to point to the last loop to be transformed to the kernel,
// and to have (numBlockDims + numThreadDims) perfectly nested loops between
// "rootForOp" and "innermostForOp".
void AffineLoopToGpuConverter::createGlobalID(AffineForOp rootForOp,
    AffineForOp innermostForOp, unsigned numBlockDims, unsigned numThreadDims) {
  OpBuilder builder(rootForOp.getOperation());
  // Prepare the grid and block sizes for the launch operation.  If there is
  // no loop mapped to a specific dimension, use constant "1" as its size.
  Value constOne =
      (numBlockDims < 3 || numThreadDims < 3)
          ? builder.create<arith::ConstantIndexOp>(rootForOp.getLoc(), 1)
          : nullptr;
  Value gridSizeX = numBlockDims > 0 ? dims[0] : constOne;
  Value gridSizeY = numBlockDims > 1 ? dims[1] : constOne;
  Value gridSizeZ = numBlockDims > 2 ? dims[2] : constOne;
  Value blockSizeX = numThreadDims > 0 ? dims[numBlockDims] : constOne;
  Value blockSizeY = numThreadDims > 1 ? dims[numBlockDims + 1] : constOne;
  Value blockSizeZ = numThreadDims > 2 ? dims[numBlockDims + 2] : constOne;

  // Create a gpu::globalID op for each loop mapped to a block or thread
  // dimension.
  //let global_idx = global_id.x;

  Type indexType = IndexType::get(rootForOp.getContext());
  Value xDim = builder.create<gpu::GlobalIdOp>(
      rootForOp.getLoc(), indexType, gpu::Dimension::x);
  // let group_size = M * N;
  Value groupSize = builder.create<arith::MulIOp>(rootForOp.getLoc(), blockSizeX, gridSizeX);

  //if (global_idx >= 9u) {
  //  return;
  //}

  //let mn = global_idx % (M * N);
  Value mn = builder.create<arith::RemSIOp>(rootForOp.getLoc(), xDim, groupSize);
  //let n = global_idx % N;
  Value n =
      builder.create<arith::RemSIOp>(rootForOp.getLoc(), xDim, gridSizeX);
  //let m = mn / N;
  Value m =
      builder.create<arith::DivSIOp>(rootForOp.getLoc(), mn, gridSizeX);

  
  Value constZero = builder.create<arith::ConstantIndexOp>(rootForOp.getLoc(), 0);
  // Build kernel dim3 values.
  KernelDim3 BlockIds = {m, constZero, constZero};
  KernelDim3 ThreadIds = {n, constZero, constZero};

  // Remap the loop iterators to use block/thread identifiers instead.  Loops
  // may iterate from LB with step S whereas GPU thread/block ids always iterate
  // from 0 to N with step 1.  Therefore, loop induction variables are replaced
  // with (gpu-thread/block-id * S) + LB.
  builder.setInsertionPoint(rootForOp);
  auto *lbArgumentIt = lbs.begin();
  auto *stepArgumentIt = steps.begin();
  for (const auto &en : llvm::enumerate(ivs)) {
    Value id =
        en.index() < numBlockDims
                   ? getDim3Value(BlockIds, en.index())
                   : getDim3Value(ThreadIds, en.index() - numBlockDims);
    Value step = steps[en.index()];
    if (getConstantIntValue(step) != static_cast<int64_t>(1))
      id = builder.create<arith::MulIOp>(rootForOp.getLoc(), step, id);

    Value ivReplacement =
        builder.create<arith::AddIOp>(rootForOp.getLoc(), *lbArgumentIt, id);
    en.value().replaceAllUsesWith(ivReplacement);
    std::advance(lbArgumentIt, 1);
    std::advance(stepArgumentIt, 1);
  }

  // create if (global_idx < group_size)
  Value cond = builder.create<arith::CmpIOp>(
      rootForOp.getLoc(), arith::CmpIPredicate::slt, xDim, groupSize);
  auto ifOp = builder.create<scf::IfOp>(rootForOp.getLoc(), ValueRange(), cond, false);
  auto &thenBlk = ifOp.getThenRegion().back();

  // Move the operations of rootForOp's body into thenBlk.
  thenBlk.getOperations().splice(thenBlk.begin(),
      innermostForOp.getBody()->getOperations());

  auto prevTerm = thenBlk.back().getPrevNode();
  if (prevTerm && isa<AffineYieldOp>(prevTerm)) {
    prevTerm->erase();
  }

  // Move allocs which only used in the thenBlk into thenBlk.
  // This will help mem2reg remove the allocs.
  auto usedInThenBlk = [&ifOp](Operation *op) {
    return llvm::all_of(op->getUsers(), [&](Operation *user) {
      auto parentIf = user->getParentOfType<scf::IfOp>();
      return parentIf == ifOp;
    });
  };

  for (auto &op :
      llvm::make_early_inc_range(rootForOp->getBlock()->getOperations())) {
    if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
      if (usedInThenBlk(allocaOp)) {
        allocaOp->moveBefore(&thenBlk, thenBlk.begin());
      }
    }
  }

  // We are done and can erase the original outermost loop.
  rootForOp.erase();
}

std::optional<AffineLoopToGpuConverter> convertAffineLoopNestToGPUGlobalID(
    AffineForOp forOp, unsigned numBlockDims = 1, unsigned numThreadDims = 1) {
  auto loops = collectAffineLoopNestBounds(forOp);
  // No top level loops.
  if (loops.empty())
    return std::nullopt;

  // Translate the loop nest to a GPU kernel in 1D.
  // 1D is easier to handle, the cost is extra instruction to compute the index
  // for each loop.
  unsigned totalLoopCount = 1;
  for (auto &loop : loops) {
    auto lb = loop.getConstantLowerBound();
    auto ub = loop.getConstantUpperBound();
    auto step = loop.getStep();
    auto range = ub - lb;
    totalLoopCount *= range / step.getZExtValue();
  }

  const unsigned MaxGroupSize = 1024;
  unsigned groupSize = MaxGroupSize;
  unsigned numDispatch = 1;
  if (totalLoopCount > MaxGroupSize) {
    // Need dispatch multiple groups.
    numDispatch = (totalLoopCount + (MaxGroupSize - 1)) / MaxGroupSize;
  } else {
    // Need dispatch one group.
    numDispatch = 1;
    groupSize = totalLoopCount;
  }

  OpBuilder builder(forOp.getOperation());
  Value id;
  if (numDispatch == 1)
    id = builder.create<gpu::ThreadIdOp>(forOp.getLoc(), gpu::Dimension::x);
  else
    id = builder.create<gpu::GlobalIdOp>(forOp.getLoc(), gpu::Dimension::x);

  // The result will be
  //  1D grid and 1D block;
  //  if (global_idx < totalLoopCount) {
  //    // loop body
  //  }
  // create if (global_idx < totalLoopCount)
  auto totalLoopCountC =
      builder.create<arith::ConstantIndexOp>(forOp.getLoc(), totalLoopCount);
  Value cond = builder.create<arith::CmpIOp>(
      forOp.getLoc(), arith::CmpIPredicate::slt, id, totalLoopCountC);
  auto ifOp =
      builder.create<scf::IfOp>(forOp.getLoc(), ValueRange(), cond, false);
  auto &thenBlk = ifOp.getThenRegion().back();

  // If numDispatch is 1, use threadID.
  // Else use globalID.
  builder.setInsertionPointToStart(&thenBlk);

  unsigned currentLoopCount = totalLoopCount;
  Value currentId = id;
  for (auto &loop : loops) {
    auto lb = loop.getConstantLowerBound();
    auto ub = loop.getConstantUpperBound();
    auto step = loop.getStep();
    auto range = ub - lb;

    unsigned loopCount = range / step.getZExtValue();
    //unsigned threadsPerLoop = currentLoopCount / loopCount;
    auto loopCountC =
        builder.create<arith::ConstantIndexOp>(forOp.getLoc(), loopCount);
    // Idx = global_idx % loopCount;
    Value idx =
        builder.create<arith::RemSIOp>(forOp.getLoc(), currentId, loopCountC);
    assert(currentLoopCount % loopCount == 0);
    // global_idx = global_idx / loopCount;
    currentId =
        builder.create<arith::DivSIOp>(forOp.getLoc(), currentId, loopCountC);

    // Replace loop induction variable with idx.
    auto lbc = builder.create<arith::ConstantIndexOp>(forOp.getLoc(), lb);
    auto ivReplacement =
        builder.create<arith::AddIOp>(forOp.getLoc(), lbc, idx);
    loop.getInductionVar().replaceAllUsesWith(ivReplacement);

    // If loop has result, we need to replace the loop with the yield value.
    if (!loop.getResultTypes().empty()) {
      if (!loop->use_empty()) {
        auto yieldOp = cast<AffineYieldOp>(loop.getBody()->getTerminator());
        auto yieldValue = yieldOp.getOperand(0);
        loop.getResult(0).replaceAllUsesWith(yieldValue);
      }
      // Replace arg with init value.
      for (auto it : llvm::zip(loop.getRegionIterArgs(), loop.getInits())) {
        auto arg = std::get<0>(it);
        auto init = std::get<1>(it);
        arg.replaceAllUsesWith(init);
      }
    }
  }

  auto innermostForOp = loops.back();
  // Move the operations of rootForOp's body into thenBlk.
  thenBlk.getOperations().splice(
      std::prev(thenBlk.end()), innermostForOp.getBody()->getOperations());

  auto prevTerm = thenBlk.back().getPrevNode();
  if (prevTerm && isa<AffineYieldOp>(prevTerm)) {
    prevTerm->erase();
  }

  // delete the loop.
  forOp.erase();

  //if (failed(checkAffineLoopNestMappable(forOp, numBlockDims, numThreadDims)))
  //  return std::nullopt;

  AffineLoopToGpuConverter converter;
  //auto maybeInnerLoop =
  //    converter.collectBounds(forOp, numBlockDims + numThreadDims);
  //if (!maybeInnerLoop)
  //  return std::nullopt;
  //converter.createGlobalID(forOp, *maybeInnerLoop, numBlockDims, numThreadDims);
  converter.NumGroups = numDispatch;
  converter.GroupSize = groupSize;
  return converter;
}

void lowerAffineLoadStoreToMemRefLoadStore(Region &body) {

  SmallVector<affine::AffineLoadOp, 4> loops;

  // Gathers all loops.
  body.walk(
      [&](affine::AffineLoadOp forOp) { loops.push_back(forOp); });

  for (auto forOp : loops) {
    OpBuilder builder(forOp.getOperation());
    auto memRefType = forOp.getMemRefType();
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(forOp.getMapOperands());
    auto resultOperands = affine::expandAffineMap(
        builder, forOp.getLoc(), forOp.getAffineMap(), indices);
    Value load = builder.create<memref::LoadOp>(
        forOp.getLoc(), forOp.getMemRef(), *resultOperands);
    forOp.replaceAllUsesWith(load);
    forOp.erase();
  }

  SmallVector<affine::AffineStoreOp, 4> stores;
  body.walk(
      [&](affine::AffineStoreOp forOp) { stores.push_back(forOp); });
  for (auto store : stores) {
    OpBuilder builder(store.getOperation());
    auto memRefType = store.getMemRefType();
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(store.getMapOperands());
    auto maybeExpandedMap = affine::expandAffineMap(
        builder, store.getLoc(), store.getAffineMap(), indices);
    builder.create<memref::StoreOp>(store.getLoc(), store.getValueToStore(),
        store.getMemRef(), *maybeExpandedMap);
    store.erase();
  }
}

ArrayRef<char> getRawData(KrnlGlobalOp &op) {
  ArrayRef<char> rawData;
  assert(op.getValue().has_value() && "Krnl Global must always have a value");
  auto value = op.getValue().value();
  llvm::TypeSwitch<Attribute>(value)
      .Case<DenseResourceElementsAttr>([&](DenseResourceElementsAttr attr) {
        auto blob =
            value.cast<DenseResourceElementsAttr>().getRawHandle().getBlob();
        assert(blob && "Expecting dense resource with a valid blob");
        rawData = blob->getData();
      })
      .Case<DenseElementsAttr>([&](DenseElementsAttr attr) {
        DenseElementsAttr denseAttr =
            value.dyn_cast_or_null<DenseElementsAttr>();
        rawData = denseAttr.getRawData();
      })
      .Default([&](Attribute attr) { return; });
  return rawData;
}

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
  entryPointFunc.getBody();

  // Lower krnl.global into constant.
  // This is a temporary solution. We should lower krnl.global into something spirv can understand.
  // ConvertGPUToSPIRV cannot lower memref.alloc.
  {
    // auto globalOps = entryPointFunc.getOps<KrnlGlobalOp>();
    //  collect all global ops.
    SmallVector<KrnlGlobalOp, 4> globalOps;
    entryPointFunc.walk(
        [&](KrnlGlobalOp globalOp) { globalOps.push_back(globalOp); });

    for (auto globalOp : globalOps) {
      OpBuilder builder(globalOp.getOperation());
      Type type = globalOp.getType();
      auto memRefTy = cast<MemRefType>(type);
      auto eltTy = memRefTy.getElementType();
      auto shapeTy = RankedTensorType::get(memRefTy.getShape(), eltTy);

      Location loc = globalOp.getLoc();
      DenseElementsAttr valueAttr =
          DenseElementsAttr::getFromRawBuffer(shapeTy, getRawData(globalOp));

      auto constantOp = builder.create<arith::ConstantOp>(loc, valueAttr);

      auto alloc = builder.create<memref::AllocOp>(loc, memRefTy);
      builder.create<memref::TensorStoreOp>(loc, constantOp, alloc);

      globalOp.replaceAllUsesWith(alloc.getResult());
      globalOp.erase();
    }
  }



  // Create GPU module.
  OpBuilder builder(context);
  builder.setInsertionPointToStart(module.getBody());
  auto kernelModule = builder.create<gpu::GPUModuleOp>(
      entryPointFunc.getLoc(), Twine(entryPointFunc.getName(), "_GPU").str());

  {
    // translate top level loops to gpu.globalID.
    // Store short loops as we walk.
    SmallVector<affine::AffineForOp, 4> loops;

    // Gathers all top level loops.
    entryPointFunc.getBody().walk([&](affine::AffineForOp forOp) {
      if (isa<func::FuncOp>(forOp->getParentOp()))
        loops.push_back(forOp);
    });

    // For each top level affine.for.
    // create a gpu func and clone into the gpu func.
    // Then scan for operand which not defined outside of affine.
    // clone these too.
    // In the end, switch tensor memref.alloc as parameter.
    // and replace the affine for with gpu launch.
    // Then remove the affine.for.
    SmallVector<Operation *, 4> clonedForOps;
    unsigned loopCount = 0;
    for (auto forOp : loops) {
      // collect used values which defined outside of affine.for.
      llvm::SetVector<Value> usedValues;
      mlir::getUsedValuesDefinedAbove(forOp.getBodyRegion(), usedValues);
      llvm::SmallVector<Value, 4> constValues;
      llvm::SmallVector<Value, 4> argValues;
      for (auto v : usedValues) {
        if (!forOp.isDefinedOutsideOfLoop(v))
          continue;
        Operation *def = v.getDefiningOp();
        if (def && isa<KrnlGlobalOp, arith::ConstantOp, arith::ConstantIndexOp,
                arith::ConstantFloatOp, arith::ConstantIntOp>(def))
          constValues.push_back(v);
        else
          argValues.emplace_back(v);
      }

      SmallVector<Type, 4> argTypes;
      for (auto v : argValues)
        argTypes.emplace_back(v.getType());
      SmallVector<Type, 1> retTypes;
      if (!forOp->use_empty())
        retTypes.emplace_back(forOp.getResult(0).getType());
      FunctionType type = FunctionType::get(context, argTypes, retTypes);
      std::string kernelName =
          entryPointFunc.getName().str() + "_operator_" + std::to_string(loopCount++);
      builder.setInsertionPointToStart(&kernelModule.getBodyRegion().back());

      auto gpuFunc =
          builder.create<gpu::GPUFuncOp>(forOp.getLoc(), kernelName, type);
      Block &newEntryBlock = gpuFunc.getBlocks().front();

      gpuFunc->setAttr(
          gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());

      // clone forOp to newFunc.
      IRMapping mapping;
      // build mapping with argValues to newFunc arguments.
      for (unsigned i = 0; i < argValues.size(); ++i) {
        mapping.map(argValues[i], gpuFunc.getArgument(i));
      }

      OpBuilder forOpBuilder(context);
      forOpBuilder.setInsertionPointToStart(&newEntryBlock);
      // clone constValues to newEntryBlock.
      for (auto v : constValues) {
        Operation *def = v.getDefiningOp();
        Operation *newOp = def->clone();
        forOpBuilder.insert(newOp);
        mapping.map(v, newOp->getResult(0));
      }

      Operation *newForOp = forOp->clone(mapping);
      forOpBuilder.insert(newForOp);

      clonedForOps.emplace_back(newForOp);
      // Add ret value if forOp result is used.
      if (!retTypes.empty())
        forOpBuilder.create<gpu::ReturnOp>(
            forOp.getLoc(), newForOp->getResult(0));

      // Convert affine.for to gpu.globalID for cloned forOp.
      auto converter =  convertAffineLoopNestToGPUGlobalID(cast<AffineForOp>(newForOp));
      assert(converter && "fail to convert affine loop to gpu globalID");

      {
        spirv::EntryPointABIAttr entryPointInfo =
            spirv::getEntryPointABIAttr(context, {(int)converter->GroupSize, 1, 1});

        gpuFunc->setAttr(spirv::getEntryPointABIAttrName(), entryPointInfo);
      }
      // Replace forOp with gpu.launch.
      builder.setInsertionPoint(forOp);

      Location loc = gpuFunc->getLoc();
      Value numGroup = builder.create<arith::ConstantIndexOp>(loc, converter->NumGroups);
      Value groupSize = builder.create<arith::ConstantIndexOp>(loc, converter->GroupSize);

      Value one =
          builder.create<arith::ConstantIndexOp>(loc, 1);

      gpu::KernelDim3 gridSize = {numGroup, one, one};
      gpu::KernelDim3 blckSize = {groupSize, one, one};
      Value none = TypedValue<::mlir::IntegerType>{};
      Value SharedMemSize = none;
      auto launchOp = builder.create<gpu::LaunchFuncOp>(forOp.getLoc(), gpuFunc,
          gridSize, blckSize, SharedMemSize, argValues,
          builder.getType<gpu::AsyncTokenType>());

      // Replace forOp with gpu.launch.
      if (!retTypes.empty()) {
        for (unsigned i=0;i<forOp.getNumResults();++i)
          forOp.getResult(i).replaceAllUsesWith(launchOp.getResult(i));
      }
      forOp.erase();

      lowerAffineLoadStoreToMemRefLoadStore(gpuFunc.getBody());
      // Add gpu.return to gpuFunc.
      builder.setInsertionPointToEnd(&gpuFunc.getBody().back());
      builder.create<gpu::ReturnOp>(loc);

      // Lower krnl.global into constant.
      {
        // auto globalOps = entryPointFunc.getOps<KrnlGlobalOp>();
        //  collect all global ops.
        SmallVector<KrnlGlobalOp, 4> globalOps;
        gpuFunc.walk(
            [&](KrnlGlobalOp globalOp) { globalOps.push_back(globalOp); });

        for (auto globalOp : globalOps) {
          OpBuilder builder(globalOp.getOperation());
          Type type = globalOp.getType();
          auto memRefTy = cast<MemRefType>(type);
          auto eltTy = memRefTy.getElementType();
          auto shapeTy = RankedTensorType::get(memRefTy.getShape(), eltTy);

          Location loc = globalOp.getLoc();
          DenseElementsAttr valueAttr = DenseElementsAttr::getFromRawBuffer(
              shapeTy, getRawData(globalOp));

          auto constantOp = builder.create<arith::ConstantOp>(loc, valueAttr);

          auto alloc = builder.create<memref::AllocOp>(loc, memRefTy);
          builder.create<memref::TensorStoreOp>(loc, constantOp, alloc);

          globalOp.replaceAllUsesWith(alloc.getResult());
          globalOp.erase();
        }
      }


    }
  }


  //{
  //  // unroll loop
  //  // Store short loops as we walk.
  //  SmallVector<affine::AffineForOp, 4> loops;

  //  // Gathers all loops.
  //  entryPointFunc.getBody().walk([&](affine::AffineForOp forOp) {
  //      loops.push_back(forOp);
  //  });
  //  for (auto forOp : loops)
  //    (void)affine::loopUnrollFull(forOp);
  //}

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

  // Create new entry point function which has return type void.
  SmallVector<Type, 4> kernelOperandTypes;
  kernelOperandTypes.reserve(numInputs + numOutputs);
  for (Value operand : entryPointFunc.getArguments()) {
    kernelOperandTypes.push_back(operand.getType());
  }
  for (Type ty : entryPointFunc.getResultTypes()) {
    kernelOperandTypes.push_back(ty);
  }
  FunctionType type = FunctionType::get(context, kernelOperandTypes, {});
  StringRef kernelName = entryPointFunc.getName();
  entryPointFunc.setName("");
  builder.setInsertionPointToStart(&module.getBodyRegion().back());

  auto outlinedFunc =
      builder.create<func::FuncOp>(entryPointFunc.getLoc(), kernelName, type);
  outlinedFunc.addEntryBlock();
  // Remove krnl.entry_point.
  entryPointOp.erase();

  // replace uses of entryPointFunc arguments with outlinedFunc arguments.
   for (unsigned i = 0; i < numInputs; ++i) {
     auto arg = entryPointFunc.getArgument(i);
     auto newArg = outlinedFunc.getArgument(i);
     arg.replaceAllUsesWith(newArg);
   }

   // Move blocks of entryPointFunc to outlinedFunc.
   auto &outlinedRegion = outlinedFunc.getFunctionBody();
   outlinedRegion.getBlocks().splice(
       outlinedRegion.end(), entryPointFunc.getBody().getBlocks());

   // merge entry block of outlinedFunc with next block.
   auto &entryBlock = outlinedRegion.front();
   auto &nextBlock = *std::next(outlinedRegion.begin());
   entryBlock.getOperations().splice(
       entryBlock.end(), nextBlock.getOperations());
   nextBlock.dropAllReferences();
   nextBlock.erase();

   // Find return op and remove it.
   auto returnOp =
       cast<func::ReturnOp>(outlinedFunc.getBody().back().getTerminator());

   builder.setInsertionPoint(returnOp);
   builder.create<func::ReturnOp>(returnOp.getLoc());

   for (int i = 0; i < numOutputs; ++i) {
     // replace return value with outlinedFunc arguments.
     auto retV = returnOp.getOperand(i);
     auto outArg = outlinedFunc.getArgument(i + numInputs);
     retV.replaceAllUsesWith(outArg);
   }
   returnOp.erase();

   entryPointFunc.erase();

   spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(module);

   // Add SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers to
   // targetAttr.
   auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_3,
       {spirv::Capability::Shader},
       {spirv::Extension::SPV_KHR_storage_buffer_storage_class,
           spirv::Extension::SPV_KHR_variable_pointers},
       context);
   targetAttr = spirv::TargetEnvAttr::get(triple,
       spirv::getDefaultResourceLimits(context), spirv::ClientAPI::Unknown,
       spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
       spirv::TargetEnvAttr::kUnknownDeviceID);

   module->setAttr(spirv::getTargetEnvAttrName(), targetAttr);
   // required by gpu.launchFuncOp.
   module->setAttr(
       gpu::GPUDialect::getContainerModuleAttrName(), builder.getUnitAttr());
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
onnx_mlir::krnl::createConvertKrnlEntryToGPUPass() {
  return std::make_unique<ConvertKrnlEntryToGPUPass>();
}
