stencil.stencil @test_stencil() -> f32 attributes {num_dimensions = 1 : index} {
    %0 = arith.constant 0.0 : f32
    return %0 : f32
}


// CHECK: func.func @eliminate_alloc_tensors
func.func @eliminate_alloc_tensors(%out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    // CHECK: %[[BUF:.*]] = tensor.extract_slice
    %buf = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    // CHECK-NEXT: %[[RES:.*]] = stencil.apply @test_stencil() outs(%[[BUF]])    
    %res = stencil.apply @test_stencil() outs(%buf) offsets [0] : (tensor<?xf32>) -> (tensor<?xf32>)    
    // CHECK-NEXT: %[[RV:.*]] = tensor.insert_slice %[[RES]]
    %rv = tensor.insert_slice %res into %out[0][%sz][1] : tensor<?xf32> into tensor<?xf32>
    return %rv : tensor<?xf32>
}