// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s



// -----

// yield in outter iterate only.


func.func @outter(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32> ) -> (memref<3x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<3x3xf32>
  %0:3 = krnl.define_loops 3
  %i = krnl.iterate(%0#0, %0#1) with (%0#0 -> %arg2 = 0 to 3, %0#1 -> %arg3 = 0 to 3, %0#2 -> %arg4 = 0 to 4) iter_args(%arg5 = %cst) -> (f32){
    %1:2 = krnl.get_induction_var_value(%0#0, %0#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.iterate(%0#2) with () {
      %3 = krnl.get_induction_var_value(%0#2) : (!krnl.loop) -> index
      %4 = krnl.load %arg0[%1#0, %3] : memref<3x4xf32>
      %5 = krnl.load %arg1[%3, %1#1] : memref<4x3xf32>
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %arg5, %6 : f32
      krnl.store %7, %alloc[%1#0, %1#1] : memref<3x3xf32>
      krnl.yield
    }
    %8 = krnl.load %alloc[%1#0, %1#1] : memref<3x3xf32>
    %9 = arith.addf %arg5, %8 : f32
    krnl.yield %9 : f32
  }
  return %alloc : memref<3x3xf32>
}

