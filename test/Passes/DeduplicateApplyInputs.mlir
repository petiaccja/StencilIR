// CHECK: stencil.stencil private @stencil_proc_[[NR:[0-9]+]](%[[A:.*]]: tensor<?xf32>, %[[C:.*]]: tensor<?xf32>)
stencil.stencil private @stencil(%a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    // CHECK-NEXT: %[[IDX:.*]] = index
    %idx = index : vector<1xindex>
    // CHECK-NEXT: %[[SA1:.*]] = sample %[[A]][%[[IDX]]]
    // CHECK-NEXT: %[[SA2:.*]] = sample %[[A]][%[[IDX]]]
    // CHECK-NEXT: %[[SC:.*]] = sample %[[C]][%[[IDX]]]
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    %bs = sample %b[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    %cs = sample %c[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    // CHECK-NEXT: %[[V0:.*]] = arith.addf %[[SA1]], %[[SA2]]
    // CHECK-NEXT: %[[R:.*]] = arith.addf %[[V0]], %[[SC]]
    %0 = arith.addf %as, %bs : f32
    %r = arith.addf %0, %cs : f32
    // CHECK-NEXT: return %[[R]]
    return %r : f32
}

//------------------------------------------------------------------------------
// Fuse extract slice with static offsets
//------------------------------------------------------------------------------

// CHECK: func.func @offseted
func.func @offseted(%a: tensor<?xf32>, %c: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    // CHECK-NOT: stencil.apply @stencil
    // CHECK: stencil.apply @stencil_proc_[[NR]]
    %1 = stencil.apply @stencil(%a, %a, %c) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %1 : tensor<?xf32>
}