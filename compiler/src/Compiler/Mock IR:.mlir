
Mock IR:

module {
  mock.kernel @kernel_fun(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    "mock.kernel_return"(%0) : (f32) -> ()
  }
  func.func @main(%arg0: f32, %arg1: f32, %arg2: memref<?xf32>, %arg3: index, %arg4: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [%arg3, %arg4], strides: [%c1, %arg3] : memref<?xf32> to memref<?x?xf32>
    mock.kernel_call @kernel_fun<%arg3, %arg4> ((%arg0, %arg1) -> (%0)) : (f32, f32) -> memref<?x?xf32>
    return
  }
}


Mixed IR:

module {
  func.func @kernel_fun(%arg0: memref<3xindex>, %arg1: f32, %arg2: f32, %arg3: memref<?x?xf32>) {
    %0 = arith.addf %arg1, %arg2 : f32
    %c0 = arith.constant 0 : index
    %1 = memref.load %arg0[%c0] : memref<3xindex>
    %c1 = arith.constant 1 : index
    %2 = memref.load %arg0[%c1] : memref<3xindex>
    %3 = arith.index_cast %1 : index to i64
    %4 = arith.sitofp %3 : i64 to f32
    memref.store %4, %arg3[%1, %2] : memref<?x?xf32>
    return
  }
  func.func @main(%arg0: f32, %arg1: f32, %arg2: memref<?xf32>, %arg3: index, %arg4: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [%arg3, %arg4], strides: [%c1, %arg3] : memref<?xf32> to memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    affine.for %arg5 = 0 to %arg3 {
      affine.for %arg6 = 0 to %arg4 {
        %c1_2 = arith.constant 1 : index
        %1 = memref.alloca() : memref<3xindex>
        %c0_3 = arith.constant 0 : index
        memref.store %arg5, %1[%c0_3] : memref<3xindex>
        %c1_4 = arith.constant 1 : index
        memref.store %arg6, %1[%c1_4] : memref<3xindex>
        %c2 = arith.constant 2 : index
        memref.store %c1_2, %1[%c2] : memref<3xindex>
        func.call @kernel_fun(%1, %arg0, %arg1, %0) : (memref<3xindex>, f32, f32, memref<?x?xf32>) -> ()
      }
    }
    return
  }
}


Pre-lowered IR:

module {
  func.func @kernel_fun(%arg0: memref<3xindex>, %arg1: f32, %arg2: f32, %arg3: memref<?x?xf32>) {
    %0 = arith.addf %arg1, %arg2 : f32
    %c0 = arith.constant 0 : index
    %1 = memref.load %arg0[%c0] : memref<3xindex>
    %c1 = arith.constant 1 : index
    %2 = memref.load %arg0[%c1] : memref<3xindex>
    %3 = arith.index_cast %1 : index to i64
    %4 = arith.sitofp %3 : i64 to f32
    memref.store %4, %arg3[%1, %2] : memref<?x?xf32>
    return
  }
  func.func @main(%arg0: f32, %arg1: f32, %arg2: memref<?xf32>, %arg3: index, %arg4: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [%arg3, %arg4], strides: [%c1, %arg3] : memref<?xf32> to memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    cf.br ^bb1(%c0_2 : index)
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb5
    %2 = arith.cmpi slt, %1, %arg3 : index
    cf.cond_br %2, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %c0_4 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    cf.br ^bb3(%c0_4 : index)
  ^bb3(%3: index):  // 2 preds: ^bb2, ^bb4
    %4 = arith.cmpi slt, %3, %arg4 : index
    cf.cond_br %4, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %c1_6 = arith.constant 1 : index
    %5 = memref.alloca() : memref<3xindex>
    %c0_7 = arith.constant 0 : index
    memref.store %1, %5[%c0_7] : memref<3xindex>
    %c1_8 = arith.constant 1 : index
    memref.store %3, %5[%c1_8] : memref<3xindex>
    %c2 = arith.constant 2 : index
    memref.store %c1_6, %5[%c2] : memref<3xindex>
    call @kernel_fun(%5, %arg0, %arg1, %0) : (memref<3xindex>, f32, f32, memref<?x?xf32>) -> ()
    %6 = arith.addi %3, %c1_5 : index
    cf.br ^bb3(%6 : index)
  ^bb5:  // pred: ^bb3
    %7 = arith.addi %1, %c1_3 : index
    cf.br ^bb1(%7 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}


LLVM IR:

module {
  llvm.func @kernel_fun(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: f32, %arg6: f32, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg7, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.insertvalue %arg8, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg10, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg12, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg11, %11[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg13, %12[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.fadd %arg5, %arg6  : f32
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.getelementptr %16[%15] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %18 = llvm.load %17 : !llvm.ptr<i64>
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %22 = llvm.load %21 : !llvm.ptr<i64>
    %23 = llvm.sitofp %18 : i64 to f32
    %24 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.mul %18, %25  : i64
    %27 = llvm.add %26, %22  : i64
    %28 = llvm.getelementptr %24[%27] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %23, %28 : !llvm.ptr<f32>
    llvm.return
  }
  llvm.func @main(%arg0: f32, %arg1: f32, %arg2: !llvm.ptr<f32>, %arg3: !llvm.ptr<f32>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) {
    llvm.call @kernel_fun(%51, %52, %53, %54, %55, %arg0, %arg1, %56, %57, %58, %arg7, %arg8, %7, %arg7) : (!llvm.ptr<i64>, !llvm.ptr<i64>, i64, i64, i64, f32, f32, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> ()


    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg4, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg5, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %5[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %9, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %10, %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %6, %12[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg7, %13[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %7, %14[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %arg8, %15[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg7, %16[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb5
    %23 = llvm.icmp "slt" %22, %arg7 : i64
    llvm.cond_br %23, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%24 : i64)
  ^bb3(%26: i64):  // 2 preds: ^bb2, ^bb4
    %27 = llvm.icmp "slt" %26, %arg8 : i64
    llvm.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.mlir.constant(3 : index) : i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.null : !llvm.ptr<i64>
    %32 = llvm.getelementptr %31[%29] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %33 = llvm.ptrtoint %32 : !llvm.ptr<i64> to i64
    %34 = llvm.alloca %33 x i64 : (i64) -> !llvm.ptr<i64>
    %35 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.insertvalue %38, %37[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.insertvalue %29, %39[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.insertvalue %30, %40[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.getelementptr %43[%42] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %22, %44 : !llvm.ptr<i64>
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %47 = llvm.getelementptr %46[%45] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %26, %47 : !llvm.ptr<i64>
    %48 = llvm.mlir.constant(2 : index) : i64
    %49 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.getelementptr %49[%48] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %28, %50 : !llvm.ptr<i64>
    %51 = llvm.extractvalue %41[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.extractvalue %41[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.extractvalue %41[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.extractvalue %17[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %57 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.extractvalue %17[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.extractvalue %17[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.extractvalue %17[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

    %14 = llvm.insertvalue %arg7, %13[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %7, %14[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %arg8, %15[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg7, %16[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

    %61 = llvm.extractvalue %17[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.extractvalue %17[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @kernel_fun(%51, %52, %53, %54, %55, %arg0, %arg1, %56, %57, %58, %59, %60, %61, %62) : (!llvm.ptr<i64>, !llvm.ptr<i64>, i64, i64, i64, f32, f32, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> ()
    %63 = llvm.add %26, %25  : i64
    llvm.br ^bb3(%63 : i64)
  ^bb5:  // pred: ^bb3
    %64 = llvm.add %22, %21  : i64
    llvm.br ^bb1(%64 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 
17 17 17 17 17 17 17 17 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 