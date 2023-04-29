// CHECK: func.func @test_stencil(%[[IDX:.*]]: vector<1xindex>, %[[ARG:.*]]: tensor<?xf32>) -> f32
stencil.stencil @test_stencil(%arg: tensor<?xf32>) -> f32 attributes {num_dimensions = 1 : i64} {
    %idx = index : vector<1xindex>
    // CHECK-NEXT: %[[V:.*]] = stencil.sample %[[ARG]][%[[IDX]]]
    %v = sample %arg[%idx] : (tensor<?xf32>, vector<1xindex>) -> f32
    return %v : f32
}


// CHECK: func.func @lower_apply_1d(%[[ARG:.*]]: tensor<?xf32>, %[[OUT:.*]]: tensor<?xf32>)
func.func @lower_apply_1d(%arg: tensor<?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
    // CHECK-NEXT: %[[V:.*]] = stencil.generate outs(%[[OUT]]) offsets [0] : (tensor<?xf32>) -> tensor<?xf32> {
    // CHECK-NEXT: ^bb0(%[[IDX:.*]]: vector<1xindex>):
    // CHECK-NEXT: %[[V_BLK:.*]] = func.call @test_stencil(%[[IDX]], %[[ARG]]) : (vector<1xindex>, tensor<?xf32>) -> f32
    // CHECK-NEXT: stencil.yield %[[V_BLK]] : f32
    // CHECK-NEXT: }
    // CHECK-NEXT: return %[[V]] : tensor<?xf32>
    %v = stencil.apply @test_stencil(%arg) outs(%out) offsets [0] : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    return %v : tensor<?xf32>
}