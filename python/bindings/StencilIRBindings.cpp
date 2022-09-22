#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/ASTNodes.hpp>

using namespace ast;


PYBIND11_MODULE(stencilir_bindings, m) {
    m.doc() = "Stencil IR Python bindings";

    pybind11::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def(pybind11::init<std::vector<std::shared_ptr<Function>>,
                            std::vector<std::shared_ptr<Stencil>>,
                            std::optional<Location>>());

    pybind11::class_<Function, std::shared_ptr<Function>>(m, "Function")
        .def(pybind11::init<std::string,
                            std::vector<Parameter>,
                            std::vector<types::Type>,
                            std::vector<std::shared_ptr<Statement>>,
                            std::optional<Location>>());

    pybind11::class_<Stencil, std::shared_ptr<Stencil>>(m, "Stencil")
        .def(pybind11::init<std::string,
                            std::vector<Parameter>,
                            std::vector<types::Type>,
                            std::vector<std::shared_ptr<Statement>>,
                            size_t,
                            std::optional<Location>>());

    pybind11::class_<Parameter>(m, "Parameter")
        .def(pybind11::init<std::string,
                            types::Type>());

    pybind11::class_<Location>(m, "Location")
        .def(pybind11::init<std::string,
                            int,
                            int>());

    pybind11::class_<Statement, std::shared_ptr<Statement>> statement(m, "Statement");

    pybind11::class_<types::FundamentalType> fundamentalType(m, "FundamentalType");
    pybind11::enum_<types::FundamentalType::eType>(m, "FundamentalTypeValues")
        .value("SINT8", types::FundamentalType::SINT8)
        .value("SINT16", types::FundamentalType::SINT16)
        .value("SINT32", types::FundamentalType::SINT32)
        .value("SINT64", types::FundamentalType::SINT64)
        .value("UINT8", types::FundamentalType::UINT8)
        .value("UINT16", types::FundamentalType::UINT16)
        .value("UINT32", types::FundamentalType::UINT32)
        .value("UINT64", types::FundamentalType::UINT64)
        .value("SSIZE", types::FundamentalType::SSIZE)
        .value("USIZE", types::FundamentalType::USIZE)
        .value("FLOAT32", types::FundamentalType::FLOAT32)
        .value("FLOAT64", types::FundamentalType::FLOAT64)
        .value("BOOL", types::FundamentalType::BOOL)
        .export_values();
    pybind11::class_<types::Type> type(m, "Type");
}