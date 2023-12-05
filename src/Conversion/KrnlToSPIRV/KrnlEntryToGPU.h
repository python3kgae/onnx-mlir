//===- KernlEntryToGPU.h - Krnl entry to GPU Patterns ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Krnl dialect entry to GPU dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class TypeConverter;
class RewritePatternSet;

/// Appends to a pattern list additional patterns for translating Func ops
/// to SPIR-V ops. Also adds the patterns to legalize ops not directly
/// translated to SPIR-V dialect.
void populateKrnlEntryToGPUPatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns);

} // namespace mlir
