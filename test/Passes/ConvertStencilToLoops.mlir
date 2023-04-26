func.func @test_stencil(%idx: vector<1xindex>, %arg: memref<?xf32>) -> f32 attributes {num_dimensions = 1 : i64} {
    %v = stencil.sample %arg[%idx] : (memref<?xf32>, vector<1xindex>) -> f32
    return %v : f32
}


// CHECK: func.func @lower_apply_1d
func.func @lower_apply_1d(%arg: memref<?xf32>, %out: memref<?xf32>) {
    // CHECK: %[[UB:.*]] = memref.dim
    // CHECK: scf.parallel (%[[IV:.*]]) = (%{{.*}}) to (%[[UB]]) step (%{{.*}}) {
    // CHECK: %[[RESULT:.*]] = func.call @test_stencil
    stencil.generate outs(%out) offsets [0] : (memref<?xf32>) -> () {
    ^bb0(%idx: vector<1xindex>):
        %v = func.call @test_stencil(%idx, %arg) : (vector<1xindex>, memref<?xf32>) -> f32
        stencil.yield %v : f32
    }
    return
}