stencil.stencil @test_stencil_1d() -> f32 attributes {num_dimensions = 1 : index} {
    %0 = arith.constant 0.0 : f32
    return %0 : f32
}

stencil.stencil @test_stencil_2d() -> f32 attributes {num_dimensions = 2 : index} {
    %0 = arith.constant 0.0 : f32
    return %0 : f32
}

// CHECK: func.func @eliminate_alloc_tensors_1d
func.func @eliminate_alloc_tensors_1d(%out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    // CHECK: %[[BUF:.*]] = bufferization.alloc_tensor
    // CHECK-NOT: %[[SLC:.*]] = tensor.extract_slice
    %buf = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    %slc = tensor.extract_slice %buf[0][%sz][1] : tensor<?xf32> to tensor<?xf32>
    // CHECK-NEXT: %[[RES:.*]] = stencil.apply @test_stencil_1d() outs(%[[BUF]])    
    %res = stencil.apply @test_stencil_1d() outs(%slc) offsets [0] : (tensor<?xf32>) -> (tensor<?xf32>)    
    // CHECK-NOT: %[[RV:.*]] = tensor.insert_slice
    %ins = tensor.insert_slice %res into %buf[0][%sz][1] : tensor<?xf32> into tensor<?xf32>
    // CHECK-NEXT: return %[[RES]]
    return %ins : tensor<?xf32>
}

// CHECK: func.func @eliminate_alloc_tensors_2d
func.func @eliminate_alloc_tensors_2d(%out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %s0 = tensor.dim %out, %c0 : tensor<?x?xf32>
    %s1 = tensor.dim %out, %c1 : tensor<?x?xf32>
    // CHECK: %[[BUF:.*]] = bufferization.alloc_tensor
    // CHECK-NOT: %[[SLC:.*]] = tensor.extract_slice
    %buf = bufferization.alloc_tensor(%s0, %s1) : tensor<?x?xf32>
    %slc = tensor.extract_slice %buf[0, 0][%s0, %s1][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    // CHECK-NEXT: %[[RES:.*]] = stencil.apply @test_stencil_2d() outs(%[[BUF]])    
    %res = stencil.apply @test_stencil_2d() outs(%slc) offsets [0, 0] : (tensor<?x?xf32>) -> (tensor<?x?xf32>)    
    // CHECK-NOT: %[[RV:.*]] = tensor.insert_slice
    %ins = tensor.insert_slice %res into %buf[0, 0][%s0, %s1][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    // CHECK-NEXT: return %[[RES]]
    return %ins : tensor<?x?xf32>
}