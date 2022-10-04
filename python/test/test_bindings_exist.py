import stencilir

def test_bindings_exist():
    stencil = stencilir.Stencil("stencil", [], [], [], 3, None)
    function = stencilir.Function("main", [], [], [], None)
    module = stencilir.Module([function], [stencil], None)
    assert module