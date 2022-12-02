// CHECK: func.func @lower_project(%[[INDEX:.*]]: vector<4xindex>)
func.func @lower_project(%index: vector<4xindex>) {
    // CHECK-NEXT: %[[PROJ:.*]] = vector.shuffle %[[INDEX]], %[[INDEX]] [0, 2, 3]
    %projected = stencil.project %index[0, 2, 3] : (vector<4xindex>) -> vector<3xindex>
    return
}


// CHECK: func.func @lower_extend(%[[INDEX:.*]]: vector<4xindex>, %[[VALUE:.*]]: index)
func.func @lower_extend(%index: vector<4xindex>, %value: index) {
    // CHECK-NEXT: %[[VVALUE:.*]] = vector.splat %[[VALUE]] : vector<1xindex>
    // CHECK-NEXT: %[[EXT:.*]] = vector.shuffle %[[INDEX]], %[[VVALUE]] [0, 1, 4, 2, 3]
    %projected = stencil.extend %index[2], %value : (vector<4xindex>) -> vector<5xindex>
    return
}


// CHECK: func.func @lower_exchange(%[[INDEX:.*]]: vector<4xindex>, %[[VALUE:.*]]: index)
func.func @lower_exchange(%index: vector<4xindex>, %value: index) {
    // CHECK-NEXT: %[[POS:.*]] = arith.constant 2 : index
    // CHECK-NEXT: %[[EXCH:.*]] = vector.insertelement %[[VALUE]], %[[INDEX]][%[[POS]] : index]
    %projected = stencil.exchange %index[2], %value : (vector<4xindex>) -> vector<4xindex>
    return
}


// CHECK: func.func @lower_extract(%[[INDEX:.*]]: vector<4xindex>)
func.func @lower_extract(%index: vector<4xindex>) {
    // CHECK-NEXT: %[[POS:.*]] = arith.constant 2 : index
    // CHECK-NEXT: %[[ELEM:.*]] = vector.extractelement %[[INDEX]][%[[POS]] : index]
    %projected = stencil.extract %index[2] : vector<4xindex>
    return
}