stencil.stencil @test_stencil() -> f32 attributes {num_dimensions = 1 : i64} {
    %0 = arith.constant 0.0 : f32
    return %0 : f32
}


// CHECK: func.func @lower_apply_1d
func.func @lower_apply_1d(%out: memref<?xf32>) {
    // CHECK: %[[UB:.*]] = memref.dim
    // CHECK: scf.parallel (%[[IV:.*]]) = (%{{.*}}) to (%[[UB]]) step (%{{.*}}) {
    // CHECK: %[[RESULT:.*]] = stencil.invoke @test_stencil
    stencil.apply @test_stencil() outs(%out) offsets [0] : (memref<?xf32>) -> ()
    return
}