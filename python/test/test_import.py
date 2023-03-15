import stencilir

def test_bindings_exist():
    stencil = stencilir.Stencil("stencil", [], [], [], 3, True, None)
    function = stencilir.Function("main", [], [], [], True, None)
    module = stencilir.Module([function], [stencil], None)
    assert module

def test_dag_exists():
    module = stencilir.ops.ModuleOp()
    assert module