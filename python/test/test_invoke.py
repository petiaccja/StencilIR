import stencilir as sir
from stencilir import ops
import numpy as np


def test_returns_void():
    module = ops.ModuleOp()
    function = module.add(ops.FuncOp("main", sir.FunctionType([], []), True, None))
    function.add(ops.ReturnOp([], None))

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.CompiledModule(module, compile_options)

    result = compiled_module.invoke("main")
    assert result is None


def test_returns_single():
    module = ops.ModuleOp()
    function = module.add(ops.FuncOp("main", sir.FunctionType([], [sir.FloatType(64)]), True, None))
    value = function.add(ops.ConstantOp(1.0, sir.FloatType(64), None)).get_result()
    function.add(ops.ReturnOp([value], None))

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.CompiledModule(module, compile_options)

    result = compiled_module.invoke("main")
    assert isinstance(result, float)
    assert result == 1.0


def test_returns_multiple():
    module = ops.ModuleOp()
    function = module.add(ops.FuncOp("main", sir.FunctionType([], [sir.FloatType(64), sir.FloatType(64)]), True, None))
    v1 = function.add(ops.ConstantOp(1.0, sir.FloatType(64), None)).get_result()
    v2 = function.add(ops.ConstantOp(2.0, sir.FloatType(64), None)).get_result()
    function.add(ops.ReturnOp([v1, v2], None))

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.CompiledModule(module, compile_options)

    result = compiled_module.invoke("main")
    assert isinstance(result, tuple)
    assert result[0] == 1.0
    assert result[1] == 2.0


def test_scalar_passthrough():
    module = ops.ModuleOp()
    function = module.add(ops.FuncOp("main", sir.FunctionType([sir.FloatType(64)], [sir.FloatType(64)]), True, None))
    value = function.get_region_arg(0)
    function.add(ops.ReturnOp([value], None))

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.CompiledModule(module, compile_options)

    value = 3.14
    result = compiled_module.invoke("main", value)
    assert value == result


def test_field_passthrough():
    field_t = sir.FieldType(sir.FloatType(64), 2)

    module = ops.ModuleOp()
    function = module.add(ops.FuncOp("main", sir.FunctionType([field_t], [field_t]), True, None))
    value = function.get_region_arg(0)
    function.add(ops.ReturnOp([value], None))

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O0)
    compiled_module = sir.CompiledModule(module, compile_options)

    value = np.ones((3, 4), dtype=np.float64)
    result = compiled_module.invoke("main", value)
    assert np.allclose(value, result)
    assert np.all(value.strides == result.strides)


def test_mixed_passthrough():
    scalar_t = sir.FloatType(64)
    field_t = sir.FieldType(sir.FloatType(64), 2)

    module = ops.ModuleOp()
    function = module.add(ops.FuncOp("main", sir.FunctionType([field_t, scalar_t], [field_t, scalar_t]), True, None))
    field_v = function.get_region_arg(0)
    scalar_v = function.get_region_arg(1)
    function.add(ops.ReturnOp([field_v, scalar_v], None))

    compile_options = sir.CompileOptions(sir.TargetArch.X86, sir.OptimizationLevel.O3)
    compiled_module = sir.CompiledModule(module, compile_options)

    v1, v2 = np.ones((3, 4), dtype=np.float64), 12.0
    v1r, v2r = compiled_module.invoke("main", v1, v2)
    assert np.allclose(v1, v1r)
    assert v2 == v2r
