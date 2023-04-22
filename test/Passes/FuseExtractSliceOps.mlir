// CHECK: stencil.stencil private @stencil_proc_[[NR:[0-9]+]](%[[A:.*]]: tensor<?xf32>)
stencil.stencil private @stencil(%a: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : i64} {
    // CHECK-NEXT: %[[IDX:.*]] = index
    %idx = index : vector<1xindex>
    // CHECK-NEXT: %[[OFF:.*]] = jump %[[IDX]], [3]
    // CHECK-NEXT: %[[S:.*]] = sample %[[A]][%[[OFF]]]
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    // CHECK-NEXT: return %[[S]]
    return %as : f32
}

//------------------------------------------------------------------------------
// Fuse extract slice with static offsets
//------------------------------------------------------------------------------

// CHECK: func.func @offseted
func.func @offseted(%a: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    %sze = arith.subi %sz, %c3 : index
    %0 = tensor.extract_slice %a[3][%sze][1] : tensor<?xf32> to tensor<?xf32>
    // CHECK-NOT: stencil.apply @stencil
    // CHECK: stencil.apply @stencil_proc_[[NR]]
    %1 = stencil.apply @stencil(%0) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %1 : tensor<?xf32>
}


//------------------------------------------------------------------------------
// DON'T fuse extract slice with DYNAMIC offsets
//------------------------------------------------------------------------------

// CHECK: func.func @dynamic
func.func @dynamic(%a: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    %sze = arith.subi %sz, %c3 : index
    %0 = tensor.extract_slice %a[%c3][%sze][1] : tensor<?xf32> to tensor<?xf32>
    // CHECK: stencil.apply @stencil
    // CHECK-NOT: stencil.apply @stencil_proc_
    %1 = stencil.apply @stencil(%0) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %1 : tensor<?xf32>
}


//------------------------------------------------------------------------------
// DON'T fuse extract slice with NON-ONE strides
//------------------------------------------------------------------------------

// CHECK: func.func @nonone
func.func @nonone(%a: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %sz = tensor.dim %out, %c0 : tensor<?xf32>
    %sze = arith.subi %sz, %c3 : index
    %0 = tensor.extract_slice %a[3][%sze][2] : tensor<?xf32> to tensor<?xf32>
    // CHECK: stencil.apply @stencil
    // CHECK-NOT: stencil.apply @stencil_proc_
    %1 = stencil.apply @stencil(%0) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %1 : tensor<?xf32>
}
