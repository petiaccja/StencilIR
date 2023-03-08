stencil.stencil private @stencil(%a: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : index} {
    %idx = index : vector<1xindex>
    %as = sample %a[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    return %as : f32
}

//------------------------------------------------------------------------------
// Fuse extract slice with static offsets
//------------------------------------------------------------------------------

// CHECK: func.func @tensor_dim_value(%[[a:.*]]: tensor<?xf32>, %[[OUT:.*]]: tensor<?xf32>)
func.func @tensor_dim_value(%a: tensor<?xf32>, %out: tensor<?xf32>) -> index {
    %1 = stencil.apply @stencil(%a) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    %c0 = arith.constant 0 : index
    // CHECK: %[[SZ:.*]] = tensor.dim %[[OUT]]
    %sz = tensor.dim %1, %c0 : tensor<?xf32>
    // CHECK: return %[[SZ]]
    return %sz : index
}
