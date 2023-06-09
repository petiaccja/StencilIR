import stencilir as sir
from stencilir import ops
import pytest
import numpy as np


def test_array_add():
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CUDA not available on the system")

    scalar_t = sir.FloatType(64)
    field_t = sir.FieldType(scalar_t, 2)

    module = ops.ModuleOp()

    stencil = module.add(ops.StencilOp("add", sir.FunctionType([field_t, field_t], [scalar_t]), 2, True, None))
    lhs = stencil.get_region_arg(0)
    rhs = stencil.get_region_arg(1)
    idx = stencil.add(ops.IndexOp(None)).get_result()
    lhs_sample = stencil.add(ops.SampleOp(lhs, idx, None)).get_result()
    rhs_sample = stencil.add(ops.SampleOp(rhs, idx, None)).get_result()
    sum = stencil.add(ops.ArithmeticOp(lhs_sample, rhs_sample, ops.ArithmeticFunction.ADD, None)).get_result()
    stencil.add(ops.ReturnOp([sum], None))

    function = module.add(ops.FuncOp("main", sir.FunctionType([field_t, field_t, field_t], [field_t]), True, None))
    lhs = function.get_region_arg(0)
    rhs = function.get_region_arg(1)
    out = function.get_region_arg(2)
    sum = function.add(ops.ApplyOp(stencil, [lhs, rhs], [out], [], [0, 0], None)).get_results()[0]
    function.add(ops.ReturnOp([sum], None))

    compile_options = sir.CompileOptions(sir.Accelerator.CUDA, sir.OptimizationLevel.O0)
    compiled_module = sir.CompiledModule(module, compile_options)

    in_lhs = cp.ones((3, 4), dtype=cp.float64)
    in_rhs = cp.ones((3, 4), dtype=cp.float64)
    in_result = cp.ones((3, 4), dtype=cp.float64)
    result = compiled_module.invoke("main", in_lhs, in_rhs, in_result)
    assert cp.allclose(in_lhs + in_rhs, in_result)
    assert np.all(in_result.strides == result.strides)