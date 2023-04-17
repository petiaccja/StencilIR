import stencilir
from stencilir import ops

def test_bindings_exist():
    module = ops.ModuleOp()
    assert module

def test_dag_exists():
    module = stencilir.ops.ModuleOp()
    assert module