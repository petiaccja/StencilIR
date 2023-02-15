//------------------------------------------------------------------------------
// Fusing a chain of pointwise stencils
//------------------------------------------------------------------------------

// CHECK: stencil.stencil private @chain_add_proc_[[NR_CHAIN:[0-9]+]](%[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>, %[[C:.*]]: tensor<?xf32>, %[[D:.*]]: tensor<?xf32>)
stencil.stencil private @chain_add(%a: tensor<?xf32>, %b: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    // CHECK-NEXT: %[[IDX:.*]] = index
    %idx = index : vector<1xindex>
    // CHECK-NEXT: %[[CS:.*]] = sample %[[C]][%[[IDX]]]
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    // CHECK-NEXT: %[[DS:.*]] = sample %[[D]][%[[IDX]]]
    %bs = sample %b[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    // CHECK-NEXT: %[[R0:.*]] = arith.addf %[[CS]], %[[DS]]
    %r = arith.addf %as, %bs : f32
    // CHECK-NEXT: %[[BS:.*]] = sample %[[B]][%[[IDX]]]
    // CHECK-NEXT: %[[R1:.*]] = arith.addf %[[R0]], %[[BS]]
    // CHECK-NEXT: %[[AS:.*]] = sample %[[A]][%[[IDX]]]
    // CHECK-NEXT: %[[R2:.*]] = arith.addf %[[R1]], %[[AS]]
    // CHECK-NEXT: return %[[R2]]
    return %r : f32
}

// CHECK: func.func @chain
func.func @chain(%a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xf32>, %d: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    %tmp0 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    %tmp1 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    // CHECK-NOT: %[[R:.*]] stencil.apply @chain_add
    // CHECK: %[[R:.*]] stencil.apply @chain_add_proc_[[NR_CHAIN]]
    %0 = stencil.apply @chain_add(%a, %b) outs(%tmp0) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    %1 = stencil.apply @chain_add(%0, %c) outs(%tmp1) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    %2 = stencil.apply @chain_add(%1, %d) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %2 : tensor<?xf32>
}


//------------------------------------------------------------------------------
// Fusing a multiple-return-valued stencil
//------------------------------------------------------------------------------

// CHECK-NOT: stencil.stencil private @mrv_source
stencil.stencil private @mrv_source(%a: tensor<?xf32>) -> (f32, f32) attributes {num_dimensions = 1 : index} {
    %idx = index : vector<1xindex>
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    return %as, %as : f32, f32
}

// CHECK: stencil.stencil private @mrv_target_proc_[[NR_MRV:[0-9]+]](%[[A:.*]]: tensor<?xf32>)
stencil.stencil private @mrv_target(%a: tensor<?xf32>, %b: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    // CHECK-NEXT: %[[IDX:.*]] = index
    %idx = index : vector<1xindex>
    // CHECK-NEXT: %[[AS:.*]] = sample %[[A]][%[[IDX]]]
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    // CHECK-NEXT: %[[BS:.*]] = sample %[[A]][%[[IDX]]]
    %bs = sample %b[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    // CHECK-NEXT: %[[R0:.*]] = arith.mulf %[[AS]], %[[BS]]
    %r = arith.mulf %as, %bs : f32
    // CHECK-NEXT: return %[[R0]]
    return %r : f32
}

// CHECK: func.func @mrv
func.func @mrv(%a: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    %tmp0 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    %tmp1 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    // CHECK-NOT: %[[R:.*]] stencil.apply @mrv_source
    // CHECK-NOT: %[[R:.*]] stencil.apply @mrv_target
    // CHECK: %[[R:.*]] stencil.apply @mrv_target_proc_[[NR_MRV]]
    %0, %1 = stencil.apply @mrv_source(%a) outs(%tmp0, %tmp1) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
    %2 = stencil.apply @mrv_target(%0, %1) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %2 : tensor<?xf32>
}


//------------------------------------------------------------------------------
// Not fusing applies with non-zero offsets
//------------------------------------------------------------------------------

// CHECK: stencil.stencil private @offseted_stencil
stencil.stencil private @offseted_stencil(%a: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    %idx = index : vector<1xindex>
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    return %as : f32
}

// CHECK: func.func @offseted
func.func @offseted(%a: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    %tmp0 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    // CHECK: %[[R:.*]] stencil.apply @offseted_stencil
    // CHECK-NEXT: %[[R:.*]] stencil.apply @offseted_stencil
    %0 = stencil.apply @offseted_stencil(%a) outs(%tmp0) offsets [1] : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    %1 = stencil.apply @offseted_stencil(%0) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %1 : tensor<?xf32>
}