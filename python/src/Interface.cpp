#include "CompiledModule.hpp"
#include <DAG/Ops.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/Nodes.hpp>
#include <Diagnostics/Exception.hpp>


using namespace ast;


CompiledModule Compile(std::shared_ptr<Module> ast, CompileOptions options, bool storeIr = false) {
    return CompiledModule{ ast, options, storeIr };
}


void SubmoduleIR(pybind11::module_& main) {
    using namespace dag;

    auto ir = main.def_submodule("ir");

    // Base classes
    pybind11::class_<Value, std::shared_ptr<Value>>(ir, "Value");
    pybind11::class_<Region, std::shared_ptr<Region>>(ir, "Region");
    pybind11::class_<Operation, std::shared_ptr<Operation>> operation(ir, "Operation");
    pybind11::class_<SingleRegion, std::shared_ptr<SingleRegion>> singleRegion(ir, "SingleRegion", operation);

    // Module structure
    pybind11::class_<ModuleOp, std::shared_ptr<ModuleOp>>(ir, "ModuleOp", singleRegion);
    pybind11::class_<FuncOp, std::shared_ptr<FuncOp>>(ir, "FuncOp", singleRegion);
    pybind11::class_<StencilOp, std::shared_ptr<StencilOp>>(ir, "StencilOp", singleRegion);
    pybind11::class_<ReturnOp, std::shared_ptr<ReturnOp>>(ir, "ReturnOp", operation);
    pybind11::class_<CallOp, std::shared_ptr<CallOp>>(ir, "CallOp", operation);
    pybind11::class_<ApplyOp, std::shared_ptr<ApplyOp>>(ir, "ApplyOp", operation);

    // Arithmetic-logic
    pybind11::class_<CastOp, std::shared_ptr<CastOp>>(ir, "CastOp", operation);
    pybind11::class_<ConstantOp, std::shared_ptr<ConstantOp>>(ir, "ConstantOp", operation);
    pybind11::class_<ArithmeticOp, std::shared_ptr<ArithmeticOp>>(ir, "ArithmeticOp", operation);
    pybind11::class_<ComparisonOp, std::shared_ptr<ComparisonOp>>(ir, "ComparisonOp", operation);
    pybind11::class_<MinOp, std::shared_ptr<MinOp>>(ir, "MinOp", operation);
    pybind11::class_<MaxOp, std::shared_ptr<MaxOp>>(ir, "MaxOp", operation);

    // Control flow
    pybind11::class_<IfOp, std::shared_ptr<IfOp>>(ir, "IfOp", operation);
    pybind11::class_<ForOp, std::shared_ptr<ForOp>>(ir, "ForOp", singleRegion);
    pybind11::class_<YieldOp, std::shared_ptr<YieldOp>>(ir, "YieldOp", operation);

    // Tensor
    pybind11::class_<DimOp, std::shared_ptr<DimOp>>(ir, "DimOp", operation);
    pybind11::class_<AllocTensorOp, std::shared_ptr<AllocTensorOp>>(ir, "AllocTensorOp", operation);
    pybind11::class_<ExtractSliceOp, std::shared_ptr<ExtractSliceOp>>(ir, "ExtractSliceOp", operation);
    pybind11::class_<InsertSliceOp, std::shared_ptr<InsertSliceOp>>(ir, "InsertSliceOp", operation);

    // Stencil
    pybind11::class_<IndexOp, std::shared_ptr<IndexOp>>(ir, "IndexOp", operation);
    pybind11::class_<JumpOp, std::shared_ptr<JumpOp>>(ir, "JumpOp", operation);
    pybind11::class_<ProjectOp, std::shared_ptr<ProjectOp>>(ir, "ProjectOp", operation);
    pybind11::class_<ExtendOp, std::shared_ptr<ExtendOp>>(ir, "ExtendOp", operation);
    pybind11::class_<ExchangeOp, std::shared_ptr<ExchangeOp>>(ir, "ExchangeOp", operation);
    pybind11::class_<ExtractOp, std::shared_ptr<ExtractOp>>(ir, "ExtractOp", operation);
    pybind11::class_<SampleOp, std::shared_ptr<SampleOp>>(ir, "SampleOp", operation);
}


PYBIND11_MODULE(stencilir, m) {
    m.doc() = "Stencil IR Python bindings";

    SubmoduleIR(m);
    //----------------------------------
    // AST nodes
    //----------------------------------
    pybind11::class_<Statement, std::shared_ptr<Statement>> statement(m, "Statement");

    pybind11::class_<Expression, std::shared_ptr<Expression>> expression(m, "Expression", statement);

    // Symbols
    pybind11::class_<SymbolRef, std::shared_ptr<SymbolRef>>(m, "SymbolRef", expression)
        .def(pybind11::init<std::string,
                            std::optional<Location>>());

    pybind11::class_<Assign, std::shared_ptr<Assign>>(m, "Assign", statement)
        .def(pybind11::init<std::vector<std::string>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>())
        .def_readwrite("names", &Assign::names)
        .def_readwrite("exprs", &Assign::exprs);

    pybind11::class_<Pack, std::shared_ptr<Pack>>(m, "Pack", expression)
        .def(pybind11::init<std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    // Stencil structure
    pybind11::class_<Stencil, std::shared_ptr<Stencil>>(m, "Stencil")
        .def(pybind11::init<std::string,
                            std::vector<Parameter>,
                            std::vector<TypePtr>,
                            std::vector<std::shared_ptr<Statement>>,
                            size_t,
                            bool,
                            std::optional<Location>>());

    pybind11::class_<Apply, std::shared_ptr<Apply>>(m, "Apply", expression)
        .def(pybind11::init<std::string,
                            std::vector<std::shared_ptr<Expression>>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::vector<int64_t>,
                            std::optional<Location>>());

    pybind11::class_<Return, std::shared_ptr<Return>>(m, "Return", statement)
        .def(pybind11::init<std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    // Module structure
    pybind11::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def(pybind11::init<std::vector<std::shared_ptr<Function>>,
                            std::vector<std::shared_ptr<Stencil>>,
                            std::optional<Location>>());

    pybind11::class_<Function, std::shared_ptr<Function>>(m, "Function")
        .def(pybind11::init<std::string,
                            std::vector<Parameter>,
                            std::vector<TypePtr>,
                            std::vector<std::shared_ptr<Statement>>,
                            bool,
                            std::optional<Location>>());

    pybind11::class_<Call, std::shared_ptr<Call>>(m, "Call", expression)
        .def(pybind11::init<std::string,
                            std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    // Stencil instrinsics
    pybind11::class_<Index, std::shared_ptr<Index>>(m, "Index", expression)
        .def(pybind11::init<std::optional<Location>>());

    pybind11::class_<Jump, std::shared_ptr<Jump>>(m, "Jump", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::vector<int64_t>,
                            std::optional<Location>>());

    pybind11::class_<Project, std::shared_ptr<Project>>(m, "Project", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::vector<int64_t>,
                            std::optional<Location>>());

    pybind11::class_<Extend, std::shared_ptr<Extend>>(m, "Extend", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            int64_t,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    pybind11::class_<Exchange, std::shared_ptr<Exchange>>(m, "Exchange", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            int64_t,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    pybind11::class_<Extract, std::shared_ptr<Extract>>(m, "Extract", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            int64_t,
                            std::optional<Location>>());

    pybind11::class_<Sample, std::shared_ptr<Sample>>(m, "Sample", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    // Control flow
    pybind11::class_<For, std::shared_ptr<For>>(m, "For", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::string,
                            std::vector<std::shared_ptr<Statement>>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::vector<std::string>,
                            std::optional<Location>>());

    pybind11::class_<If, std::shared_ptr<If>>(m, "If", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::vector<std::shared_ptr<Statement>>,
                            std::vector<std::shared_ptr<Statement>>,
                            std::optional<Location>>());

    pybind11::class_<Yield, std::shared_ptr<Yield>>(m, "Yield", expression)
        .def(pybind11::init<std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    pybind11::class_<Block, std::shared_ptr<Block>>(m, "Block", expression)
        .def(pybind11::init<std::vector<std::shared_ptr<Statement>>,
                            std::optional<Location>>());

    // Tensors
    pybind11::class_<AllocTensor, std::shared_ptr<AllocTensor>>(m, "AllocTensor", expression)
        .def(pybind11::init<TypePtr,
                            std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    pybind11::class_<Dim, std::shared_ptr<Dim>>(m, "Dim", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    pybind11::class_<ExtractSlice, std::shared_ptr<ExtractSlice>>(m, "ExtractSlice", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    pybind11::class_<InsertSlice, std::shared_ptr<InsertSlice>>(m, "InsertSlice", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    // Arithmetic-logic
    pybind11::class_<Constant, std::shared_ptr<Constant>>(m, "Constant", expression)
        .def_static("integral", [](int value, TypePtr type, std::optional<Location> loc) {
            return Constant(value, type, std::move(loc));
        })
        .def_static("floating", [](double value, TypePtr type, std::optional<Location> loc) {
            return Constant(value, type, std::move(loc));
        })
        .def_static("index", [](int64_t value, std::optional<Location> loc) {
            return Constant(value, std::make_shared<IndexType>(), std::move(loc));
        })
        .def_static("boolean", [](bool value, std::optional<Location> loc) {
            return Constant(value, std::make_shared<IntegerType>(1, true), std::move(loc));
        });

    pybind11::enum_<eArithmeticFunction>(m, "ArithmeticFunction")
        .value("ADD", eArithmeticFunction::ADD)
        .value("SUB", eArithmeticFunction::SUB)
        .value("MUL", eArithmeticFunction::MUL)
        .value("DIV", eArithmeticFunction::DIV)
        .value("MOD", eArithmeticFunction::MOD)
        .value("BIT_AND", eArithmeticFunction::BIT_AND)
        .value("BIT_OR", eArithmeticFunction::BIT_OR)
        .value("BIT_XOR", eArithmeticFunction::BIT_XOR)
        .value("BIT_SHL", eArithmeticFunction::BIT_SHL)
        .value("BIT_SHR", eArithmeticFunction::BIT_SHR)
        .export_values();

    pybind11::class_<ArithmeticOperator, std::shared_ptr<ArithmeticOperator>>(m, "ArithmeticOperator", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            eArithmeticFunction,
                            std::optional<Location>>());

    pybind11::enum_<eComparisonFunction>(m, "ComparisonFunction")
        .value("EQ", eComparisonFunction::EQ)
        .value("NEQ", eComparisonFunction::NEQ)
        .value("LT", eComparisonFunction::LT)
        .value("GT", eComparisonFunction::GT)
        .value("LTE", eComparisonFunction::LTE)
        .value("GTE", eComparisonFunction::GTE)
        .export_values();

    pybind11::class_<ComparisonOperator, std::shared_ptr<ComparisonOperator>>(m, "ComparisonOperator", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            eComparisonFunction,
                            std::optional<Location>>());

    pybind11::class_<Min, std::shared_ptr<Min>>(m, "Min", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    pybind11::class_<Max, std::shared_ptr<Max>>(m, "Max", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    pybind11::class_<Cast, std::shared_ptr<Cast>>(m, "Cast", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            TypePtr,
                            std::optional<Location>>());

    //----------------------------------
    // AST structures
    //----------------------------------
    pybind11::class_<Parameter>(m, "Parameter")
        .def(pybind11::init<std::string,
                            TypePtr>())
        .def_readwrite("name", &Parameter::name)
        .def_readwrite("type", &Parameter::type);

    pybind11::class_<Location>(m, "Location")
        .def(pybind11::init<std::string,
                            int,
                            int>())
        .def_readwrite("file", &Location::file)
        .def_readwrite("line", &Location::line)
        .def_readwrite("column", &Location::col);

    pybind11::class_<Type, std::shared_ptr<Type>> type(m, "Type");

    pybind11::class_<IntegerType, std::shared_ptr<IntegerType>>(m, "IntegerType", type)
        .def(pybind11::init<int, bool>())
        .def_readonly("size", &IntegerType::size)
        .def_readonly("is_signed", &IntegerType::isSigned);

    pybind11::class_<FloatType, std::shared_ptr<FloatType>>(m, "FloatType", type)
        .def(pybind11::init<int>())
        .def_readonly("size", &FloatType::size);

    pybind11::class_<IndexType, std::shared_ptr<IndexType>>(m, "IndexType", type)
        .def(pybind11::init<>());

    pybind11::class_<FieldType, std::shared_ptr<FieldType>>(m, "FieldType", type)
        .def(pybind11::init<TypePtr, int>())
        .def_readonly("element_type", &FieldType::elementType)
        .def_readonly("num_dimensions", &FieldType::numDimensions);


    //----------------------------------
    // Execution
    //----------------------------------
    pybind11::enum_<eTargetArch>(m, "TargetArch")
        .value("X86", eTargetArch::X86)
        .value("NVPTX", eTargetArch::NVPTX)
        .value("AMDGPU", eTargetArch::AMDGPU)
        .export_values();

    pybind11::enum_<eOptimizationLevel>(m, "OptimizationLevel")
        .value("O0", eOptimizationLevel::O0)
        .value("O1", eOptimizationLevel::O1)
        .value("O2", eOptimizationLevel::O2)
        .value("O3", eOptimizationLevel::O3)
        .export_values();

    pybind11::class_<OptimizationOptions>(m, "OptimizationOptions")
        .def(pybind11::init<bool, bool, bool, bool>(),
             pybind11::arg("eliminate_alloc_buffers") = false,
             pybind11::arg("inline_functions") = false,
             pybind11::arg("fuse_extract_slice_ops") = false,
             pybind11::arg("fuse_apply_ops") = false)
        .def_readwrite("eliminate_alloc_buffers", &OptimizationOptions::eliminateAllocBuffers)
        .def_readwrite("inline_functions", &OptimizationOptions::inlineFunctions)
        .def_readwrite("fuse_extract_slice_ops", &OptimizationOptions::fuseExtractSliceOps)
        .def_readwrite("fuse_apply_ops", &OptimizationOptions::fuseApplyOps);

    pybind11::class_<CompileOptions>(m, "CompileOptions")
        .def(pybind11::init<eTargetArch, eOptimizationLevel, OptimizationOptions>(),
             pybind11::arg("target_arch"),
             pybind11::arg("opt_level"),
             pybind11::arg("opt_options") = OptimizationOptions{})
        .def_readwrite("target_arch", &CompileOptions::targetArch)
        .def_readwrite("opt_level", &CompileOptions::optimizationLevel)
        .def_readwrite("opt_options", &CompileOptions::optimizationOptions);

    pybind11::class_<CompiledModule>(m, "CompiledModule")
        .def("invoke", &CompiledModule::Invoke)
        .def("get_ir", &CompiledModule::GetIR);

    pybind11::class_<StageResult>(m, "StageIR")
        .def_readonly("stage", &StageResult::name)
        .def_readonly("ir", &StageResult::ir);

    m.def("compile", &Compile, pybind11::arg("ast"), pybind11::arg("options"), pybind11::arg("store_ir") = false);

    //----------------------------------
    // Error handling
    //----------------------------------
    static pybind11::exception<Exception> exception(m, "Exception");
    static pybind11::exception<SyntaxError> syntaxError(m, "SyntaxError", exception);
    pybind11::register_exception<NotImplementedError>(m, "NotImplementedError", syntaxError);
    pybind11::register_exception<UndefinedSymbolError>(m, "UndefinedSymbolError", syntaxError);
    pybind11::register_exception<ArgumentTypeError>(m, "ArgumentTypeError", syntaxError);
    pybind11::register_exception<ArgumentCountError>(m, "ArgumentCountError", syntaxError);
    static pybind11::exception<CompilationError> compilationError(m, "CompilationError", syntaxError);
    pybind11::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        }
        catch (const CompilationError& ex) {
            std::stringstream ss;
            ss << ex.GetMessage();
            if (!ex.GetModule().empty()) {
                ss << std::endl
                   << "Module:"
                   << ex.GetModule();
            }
            compilationError(ss.str().c_str());
        }
    });
}