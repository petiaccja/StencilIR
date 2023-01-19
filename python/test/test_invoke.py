import stencilir as sir
import numpy as np


def test_returns_void():
    function = sir.Function(
        "main",
        [],
        [],
        [sir.Return([], None)],
        True,
        None,
    )
    module = sir.Module([function], [], None)

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.compile(module, compile_options)

    result = compiled_module.invoke("main")
    assert result is None


def test_returns_single():
    function = sir.Function(
        "main",
        [],
        [sir.FloatType(64)],
        [sir.Return([sir.Constant.floating(1.0, sir.FloatType(64), None)], None)],
        True,
        None,
    )
    module = sir.Module([function], [], None)

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.compile(module, compile_options)

    result = compiled_module.invoke("main")
    assert isinstance(result, float)
    assert result == 1.0


def test_returns_multiple():
    function = sir.Function(
        "main",
        [],
        [sir.FloatType(64), sir.FloatType(64)],
        [
            sir.Return(
                [
                    sir.Constant.floating(1.0, sir.FloatType(64), None),
                    sir.Constant.floating(2.0, sir.FloatType(64), None),
                ],
                None,
            )
        ],
        True,
        None,
    )
    module = sir.Module([function], [], None)

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.compile(module, compile_options, True)

    result = compiled_module.invoke("main")
    assert isinstance(result, tuple)
    assert result[0] == 1.0
    assert result[1] == 2.0


def test_scalar_passthrough():
    function = sir.Function(
        "main",
        [sir.Parameter("value", sir.FloatType(64))],
        [sir.FloatType(64)],
        [sir.Return([sir.SymbolRef("value", None)], None)],
        True,
        None,
    )
    module = sir.Module([function], [], None)

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.compile(module, compile_options)

    value = 3.14
    result = compiled_module.invoke("main", value)
    assert value == result


def test_field_passthrough():
    function = sir.Function(
        "main",
        [sir.Parameter("value", sir.FieldType(sir.FloatType(64), 2))],
        [sir.FieldType(sir.FloatType(64), 2)],
        [sir.Return([sir.SymbolRef("value", None)], None)],
        True,
        None,
    )
    module = sir.Module([function], [], None)

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.compile(module, compile_options, True)
    ir = compiled_module.get_ir()

    value = np.ones((3, 4), dtype=np.float64)
    result = compiled_module.invoke("main", value)
    assert np.allclose(value, result)
    assert np.all(value.strides == result.strides)


def test_mixed_passthrough():
    function = sir.Function(
        "main",
        [
            sir.Parameter("v1", sir.FieldType(sir.FloatType(64), 2)),
            sir.Parameter("v2", sir.FloatType(64)),
        ],
        [
            sir.FieldType(sir.FloatType(64), 2),
            sir.FloatType(64),
        ],
        [sir.Return([sir.SymbolRef("v1", None), sir.SymbolRef("v2", None)], None)],
        True,
        None,
    )
    module = sir.Module([function], [], None)

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3)
    compiled_module = sir.compile(module, compile_options, True)
    ir = compiled_module.get_ir()

    v1, v2 = np.ones((3, 4), dtype=np.float64), 12.0
    v1r, v2r = compiled_module.invoke("main", v1, v2)
    assert np.allclose(v1, v1r)
    assert v2 == v2r
