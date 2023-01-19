//------------------------------------------------------------------------------
// Check if two stencils are properly fused and rewritten stencil is DCE'd
//------------------------------------------------------------------------------

// CHECK: stencil.stencil private @elim_multiply
stencil.stencil private @elim_multiply(%a: tensor<?xf32>, %b: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    %idx = index : vector<1xindex>
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    %bs = sample %b[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    %r = arith.mulf %as, %bs : f32
    return %r : f32
}

// CHECK-NOT: stencil.stencil private @elim_add
// CHECK: stencil.stencil private @elim_add_fused_1(%[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>, %[[C:.*]]: tensor<?xf32>)
stencil.stencil private @elim_add(%a: tensor<?xf32>, %b: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    // CHECK: %[[IDX:.*]] = index
    %idx = index : vector<1xindex>
    // CHECK-NEXT: %[[PROD:.*]] = invoke @elim_multiply<%[[IDX]]>(%[[A]], %[[B]])
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    // CHECK-NEXT: %[[CS:.*]] = sample %[[C]][%[[IDX]]]
    %bs = sample %b[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    // CHECK-NEXT: %[[R:.*]] = arith.addf %[[PROD]], %[[CS]]
    %r = arith.addf %as, %bs : f32
    return %r : f32
}

// CHECK: func.func @elim
func.func @elim(%a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    %tmp = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    // CHECK-NOT: %[[R:.*]] stencil.apply @elim_multiply
    // CHECK-NOT: %[[R:.*]] stencil.apply @elim_add
    // CHECK: %[[R:.*]] stencil.apply @elim_add_fused_1
    %prod = stencil.apply @elim_multiply(%a, %b) outs(%tmp) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    %r = stencil.apply @elim_add(%prod, %c) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %r : tensor<?xf32>
}


//------------------------------------------------------------------------------
// Check if chain of stencils is properly fused
//------------------------------------------------------------------------------

// CHECK: stencil.stencil private @chain_add
stencil.stencil private @chain_add(%a: tensor<?xf32>, %b: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    %idx = index : vector<1xindex>
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    %bs = sample %b[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    %r = arith.addf %as, %bs : f32
    return %r : f32
}

// CHECK: func.func @chain
func.func @chain(%a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xf32>, %d: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    %tmp0 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    %tmp1 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    // CHECK-NOT: %[[R:.*]] stencil.apply @chain_add
    // CHECK-NOT: %[[R:.*]] stencil.apply @chain_add_fused_1
    // CHECK: %[[R:.*]] stencil.apply @chain_add_fused_2
    %0 = stencil.apply @chain_add(%a, %b) outs(%tmp0) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    %1 = stencil.apply @chain_add(%0, %c) outs(%tmp1) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    %2 = stencil.apply @chain_add(%1, %d) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %2 : tensor<?xf32>
}
