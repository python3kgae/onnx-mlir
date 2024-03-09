// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s


// -----

// yield in unrolled iterate.
func.func @unroll_with_block() {
  %ii = krnl.define_loops 1
  %ii1, %ii2 = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.unroll %ii2 : !krnl.loop
  krnl.iterate(%ii1) with (%ii -> %i = 0 to 8) {
    krnl.iterate(%ii2) with () {
      %i2 = krnl.get_induction_var_value(%ii2) : (!krnl.loop) -> index
      %foo = arith.addi %i2, %i2 : index
    }
  }
  return
}


// yield in unrolled iterate.

// unroll outter0, outter1, inner0, inner1, all.
func.func @unroll(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32> ) -> (memref<3x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<3x3xf32>
  %0:3 = krnl.define_loops 3
  krnl.iterate(%0#0, %0#1) with (%0#0 -> %arg2 = 0 to 3, %0#1 -> %arg3 = 0 to 3, %0#2 -> %arg4 = 0 to 4) {
    %1:2 = krnl.get_induction_var_value(%0#0, %0#1) : (!krnl.loop, !krnl.loop) -> (index, index)
    %2 = krnl.iterate(%0#2) with () iter_args(%arg5 = %cst) -> (f32){
      %3 = krnl.get_induction_var_value(%0#2) : (!krnl.loop) -> index
      %4 = krnl.load %arg0[%1#0, %3] : memref<3x4xf32>
      %5 = krnl.load %arg1[%3, %1#1] : memref<4x3xf32>
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %arg5, %6 : f32
      krnl.yield %7 : f32
    }
    krnl.store %2, %alloc[%1#0, %1#1] : memref<3x3xf32>
    krnl.yield
  }
  return %alloc : memref<3x3xf32>
}

func.func @unroll_with_block_and_permute() {
  %ii, %ij = krnl.define_loops 2
  %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.unroll %jb : !krnl.loop
  krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
    %b1, %b2 = krnl.get_induction_var_value(%ib, %jb) : (!krnl.loop, !krnl.loop) -> (index, index)
    krnl.iterate(%il, %jl) with () {
      %l1, %l2 = krnl.get_induction_var_value(%il, %jl) : (!krnl.loop, !krnl.loop) -> (index, index)
      %foo = arith.addi %l1, %l2 : index
      %bar = arith.addi %b1, %l2 : index
    }
  }
  return

  // CHECK-DAG: #map = affine_map<(d0) -> (d0)>
  // CHECK-DAG: #map1 = affine_map<(d0) -> (d0 + 5)>
  // CHECK-DAG: #map2 = affine_map<(d0) -> (d0 + 1)>
  // CHECK-DAG: #map3 = affine_map<(d0) -> (d0 + 2)>
  // CHECK-DAG: #map4 = affine_map<(d0) -> (d0 + 3)>
  // CHECK-LABEL:  unroll_with_block_and_permute
  // CHECK:        affine.for [[I_0_:%.+]] = 0 to 10 step 5 {
  // CHECK:          affine.for [[I_1_:%.+]] = 0 to 20 step 4 {
  // CHECK:            affine.for [[I_2_:%.+]] = #map([[I_0_]]) to #map1([[I_0_]]) {
  // CHECK-NEXT:         [[VAR_0_:%.+]] = arith.addi [[I_2_]], [[I_1_]] : index
  // CHECK-NEXT:         [[VAR_1_:%.+]] = arith.addi [[I_0_]], [[I_1_]] : index
  // CHECK-NEXT:         [[VAR_2_:%.+]] = affine.apply #map2([[I_1_]])
  // CHECK-NEXT:         [[VAR_3_:%.+]] = arith.addi [[I_2_]], [[VAR_2_]] : index
  // CHECK-NEXT:         [[VAR_4_:%.+]] = arith.addi [[I_0_]], [[VAR_2_]] : index
  // CHECK-NEXT:         [[VAR_5_:%.+]] = affine.apply #map3([[I_1_]])
  // CHECK-NEXT:         [[VAR_6_:%.+]] = arith.addi [[I_2_]], [[VAR_5_]] : index
  // CHECK-NEXT:         [[VAR_7_:%.+]] = arith.addi [[I_0_]], [[VAR_5_]] : index
  // CHECK-NEXT:         [[VAR_8_:%.+]] = affine.apply #map4([[I_1_]])
  // CHECK-NEXT:         [[VAR_9_:%.+]] = arith.addi [[I_2_]], [[VAR_8_]] : index
  // CHECK-NEXT:         [[VAR_10_:%.+]] = arith.addi [[I_0_]], [[VAR_8_]] : index
  // CHECK:            }
  // CHECK:          }
  // CHECK:        }
}
