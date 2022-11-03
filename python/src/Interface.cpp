#include "CompiledModule.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/Nodes.hpp>


using namespace ast;


CompiledModule Compile(std::shared_ptr<Module> ast, CompileOptions options, bool storeIr = false) {
    return CompiledModule{ ast, options, storeIr };
}


PYBIND11_MODULE(stencilir, m) {
    m.doc() = "Stencil IR Python bindings";

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
                            std::optional<Location>>());

    pybind11::class_<Pack, std::shared_ptr<Pack>>(m, "Pack", expression)
        .def(pybind11::init<std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    // Stencil structure
    pybind11::class_<Stencil, std::shared_ptr<Stencil>>(m, "Stencil")
        .def(pybind11::init<std::string,
                            std::vector<Parameter>,
                            std::vector<Type>,
                            std::vector<std::shared_ptr<Statement>>,
                            size_t,
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
                            std::vector<Type>,
                            std::vector<std::shared_ptr<Statement>>,
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

    pybind11::class_<Sample, std::shared_ptr<Sample>>(m, "Sample", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    pybind11::class_<JumpIndirect, std::shared_ptr<JumpIndirect>>(m, "JumpIndirect", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            int64_t,
                            std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    pybind11::class_<SampleIndirect, std::shared_ptr<SampleIndirect>>(m, "SampleIndirect", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            int64_t,
                            std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            std::optional<Location>>());

    // Control flow
    pybind11::class_<For, std::shared_ptr<For>>(m, "For", expression)
        .def(pybind11::init<std::shared_ptr<Expression>,
                            std::shared_ptr<Expression>,
                            int64_t,
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

    // Tensors
    pybind11::class_<AllocTensor, std::shared_ptr<AllocTensor>>(m, "AllocTensor", expression)
        .def(pybind11::init<ScalarType,
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
        .def_static("integral", [](int value, ScalarType type, std::optional<Location> loc) {
            switch (type) {
                case ScalarType::SINT8: return Constant(int8_t(value));
                case ScalarType::SINT16: return Constant(int16_t(value));
                case ScalarType::SINT32: return Constant(int32_t(value));
                case ScalarType::SINT64: return Constant(int64_t(value));
                case ScalarType::UINT8: return Constant(uint8_t(value));
                case ScalarType::UINT16: return Constant(uint16_t(value));
                case ScalarType::UINT32: return Constant(uint32_t(value));
                case ScalarType::UINT64: return Constant(uint64_t(value));
                default: throw std::invalid_argument("Provide an integer type.");
            }
        })
        .def_static("floating", [](double value, ScalarType type, std::optional<Location> loc) {
            switch (type) {
                case ScalarType::FLOAT32: return Constant(float(value));
                case ScalarType::FLOAT64: return Constant(double(value));
                default: throw std::invalid_argument("Provide an floating point type.");
            }
        })
        .def_static("index", [](ptrdiff_t value, std::optional<Location> loc) {
            return Constant(index_type, value);
        })
        .def_static("boolean", [](bool value, std::optional<Location> loc) {
            return Constant(value);
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
                            Type>());

    //----------------------------------
    // AST structures
    //----------------------------------
    pybind11::class_<Parameter>(m, "Parameter")
        .def(pybind11::init<std::string,
                            Type>());

    pybind11::class_<Location>(m, "Location")
        .def(pybind11::init<std::string,
                            int,
                            int>());

    pybind11::enum_<ScalarType>(m, "ScalarType")
        .value("SINT8", ScalarType::SINT8)
        .value("SINT16", ScalarType::SINT16)
        .value("SINT32", ScalarType::SINT32)
        .value("SINT64", ScalarType::SINT64)
        .value("UINT8", ScalarType::UINT8)
        .value("UINT16", ScalarType::UINT16)
        .value("UINT32", ScalarType::UINT32)
        .value("UINT64", ScalarType::UINT64)
        .value("INDEX", ScalarType::INDEX)
        .value("FLOAT32", ScalarType::FLOAT32)
        .value("FLOAT64", ScalarType::FLOAT64)
        .value("BOOL", ScalarType::BOOL)
        .export_values();

    pybind11::class_<FieldType>(m, "FieldType")
        .def(pybind11::init<ScalarType, size_t>());

    pybind11::class_<Type> type(m, "Type");

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

    pybind11::class_<CompileOptions>(m, "CompileOptions")
        .def(pybind11::init())
        .def(pybind11::init<eTargetArch, eOptimizationLevel>())
        .def_readwrite("target_arch", &CompileOptions::targetArch)
        .def_readwrite("opt_level", &CompileOptions::optimizationLevel);

    pybind11::class_<CompiledModule>(m, "CompiledModule")
        .def("invoke", &CompiledModule::Invoke)
        .def("get_ir", &CompiledModule::GetIR);

    pybind11::class_<StageResult>(m, "StageIR")
        .def_readonly("stage", &StageResult::name)
        .def_readonly("ir", &StageResult::ir);

    m.def("compile", &Compile, pybind11::arg("ast"), pybind11::arg("options"), pybind11::arg("store_ir") = false);
}