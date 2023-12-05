//===- FuncToSPIRV.cpp - Func to SPIR-V Patterns ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Func dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "KrnlEntryToGPU.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl-entry-to-gpu-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {

/// Converts func.return to spirv.Return.
class ReturnOpPattern final : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (returnOp.getNumOperands() > 1)
      return failure();

    rewriter.create<gpu::ReturnOp>(returnOp.getLoc());
    return success();
  }
};

/// Converts Affine to GPU.block_id.
class AffineOpPattern final : public OpConversionPattern<affine::AffineForOp> {
public:
  using OpConversionPattern<affine::AffineForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(affine::AffineForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *block = forOp->getBlock();
    if (!block)
      return failure();
    if (Region *region = block->getParent()) {
      if (!isa<gpu::GPUFuncOp>(region->getParentOp()))
        return failure();
    }

    Location loc = forOp.getLoc();
    Value lowerBound = lowerAffineLowerBound(forOp, rewriter);
    Value upperBound = lowerAffineUpperBound(forOp, rewriter);
    Value step =
        rewriter.create<arith::ConstantIndexOp>(loc, forOp.getStepAsInt());
    //if (callOp.getNumResults() == 1) {
    //  auto resultType =
    //      getTypeConverter()->convertType(callOp.getResult(0).getType());
    //  if (!resultType)
    //    return failure();
    //  rewriter.replaceOpWithNewOp<spirv::FunctionCallOp>(
    //      callOp, resultType, adaptor.getOperands(), callOp->getAttrs());
    //} else {
    //  rewriter.replaceOpWithNewOp<spirv::FunctionCallOp>(
    //      callOp, TypeRange(), adaptor.getOperands(), callOp->getAttrs());
    //}
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateKrnlEntryToGPUPatterns(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<ReturnOpPattern, AffineOpPattern>(typeConverter, context);
}

