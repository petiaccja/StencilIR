import stencilir as ir
import numpy as np


def test_exec_stencil():
    c = ir.SymbolRef("c", None)
    out = ir.SymbolRef("out", None)
    stencil = ir.Stencil(
        "fill",
        [ir.Parameter("c", ir.ScalarType.FLOAT32)],
        [ir.ScalarType.FLOAT32],
        [ir.Return([c], None)],
        2,
        None,
    )
    function = ir.Function(
        "main",
        [
            ir.Parameter("c", ir.ScalarType.FLOAT32),
            ir.Parameter("out", ir.FieldType(ir.ScalarType.FLOAT32, 2)),
        ],
        [],
        [ir.Apply("fill", [c], [out], [], [0, 0], None), ir.Return([], None)],
        None,
    )
    module = ir.Module([function], [stencil], None)

    compile_options = ir.CompileOptions(ir.TargetArch.X86, ir.OptimizationLevel.O0)

    compiled_module = ir.compile(module, compile_options)
    value = 3.14
    out = np.zeros((8, 6), dtype=np.float32)
    compiled_module.invoke("main", value, out)
    assert np.allclose(out, value)
