#include "CompiledModule.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/Nodes.hpp>


using namespace ast;


CompiledModule Compile(std::shared_ptr<Module> ast, CompileOptions options) {
    return CompiledModule{ ast, options };
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
                            std::shared_ptr<Expression>,
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
        .def("invoke", &CompiledModule::Invoke);

    m.def("compile", &Compile);
}