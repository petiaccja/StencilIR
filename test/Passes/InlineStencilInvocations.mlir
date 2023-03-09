// CHECK-NOT: stencil.stencil private @callee
stencil.stencil private @callee(%a: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    %idx = index : vector<1xindex>
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    return %as : f32
}

// CHECK: stencil.stencil public @caller(%[[A:.*]]: tensor<?xf32>)
stencil.stencil public @caller(%a: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    // CHECK: %[[IDX:.*]] = index
    %idx = index : vector<1xindex>
    // CHECK-NEXT: %[[IDX_LEFT:.*]] = jump
    %idx_left = jump %idx, [-1] : (vector<1xindex>) -> vector<1xindex>
    // CHECK-NEXT: %[[AS:.*]] = sample %[[A]][%[[IDX_LEFT]]]
    %as = invoke @callee<%idx_left>(%a) : (vector<1xindex>, tensor<?xf32>) -> f32
    // CHECK-NEXT: return %[[AS]]
    return %as : f32
}
