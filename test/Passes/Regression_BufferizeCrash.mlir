// Ensure it does not crash bufferization pass. The issue was in GenericOp/bufferize() interface method.
module {
  func.func private @work(%arg0: vector<2xindex>, %arg1: tensor<?x?xf32>) -> f32 {
    %0 = stencil.sample %arg1[%arg0] : (tensor<?x?xf32>, vector<2xindex>) -> f32
    return %0 : f32
  }
  // CHECK: func @main
  func.func @main(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %2 = tensor.empty(%arg1, %arg2) : tensor<?x?xf32>
    %3 = stencil.generate outs(%2) offsets [0, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32> {
    ^bb0(%arg4: vector<2xindex>):
      %4 = func.call @work(%arg4, %arg0) : (vector<2xindex>, tensor<?x?xf32>) -> f32
      stencil.yield %4 : f32
    }
    %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0] [%arg1, %arg2] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    return %inserted_slice : tensor<?x?xf32>
  }
}