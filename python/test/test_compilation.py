import stencilir

def test_exec_stencil():
    ret = stencilir.Return([], None)
    stencil = stencilir.Stencil("stencil", [], [], [ret], 3, None)
    function = stencilir.Function("main", [], [], [ret], None)
    module = stencilir.Module([function], [stencil], None)

    compile_options = stencilir.CompileOptions(
        stencilir.TargetArch.X86, stencilir.OptimizationLevel.O0
    )

    compiled_module = stencilir.compile(module, compile_options)
    compiled_module.invoke("main")
