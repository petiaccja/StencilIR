#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/ASTBuilding.hpp>


inline auto apply_dynamic(std::string callee,
                  std::vector<std::shared_ptr<ast::Expression>> inputs,
                  std::vector<std::shared_ptr<ast::Expression>> outputs,
                  std::vector<std::shared_ptr<ast::Expression>> offsets,
                  std::optional<ast::Location> loc = {}) {
    return std::make_shared<ast::Apply>(callee, inputs, outputs, offsets, loc);
}

inline auto apply_static(std::string callee,
                  std::vector<std::shared_ptr<ast::Expression>> inputs,
                  std::vector<std::shared_ptr<ast::Expression>> outputs,
                  std::vector<int64_t> static_offsets = {},
                  std::optional<ast::Location> loc = {}) {
    return std::make_shared<ast::Apply>(callee, inputs, outputs, static_offsets, loc);
}


PYBIND11_MODULE(stencilir_ext, m) {
    m.doc() = "Stencil IR Python bindings";

    m.def("symref", &ast::symref, "");
    m.def("assign", &ast::assign, "");
    m.def("stencil", &ast::stencil, "");
    m.def("return_", &ast::return_, "");
    m.def("apply_dynamic", &apply_dynamic, "");
    m.def("apply_static", &apply_static, "");
    m.def("function", &ast::function, "");
    m.def("module", &ast::module_, "");
    m.def("index", &ast::index, "");
    m.def("jump", &ast::jump, "");
    m.def("sample", &ast::sample, "");
    m.def("jump_indirect", &ast::jump_indirect, "");
    m.def("sample_indirect", &ast::sample_indirect, "");
    m.def("for_", &ast::for_, "");
    m.def("if_", &ast::if_, "");
    m.def("yield_", &ast::yield, "");
    //m.def("constant", &ast::constant, "");
    m.def("add", &ast::add, "");
    m.def("sub", &ast::sub, "");
    m.def("mul", &ast::mul, "");
    m.def("div", &ast::div, "");
    m.def("mod", &ast::mod, "");
    m.def("bit_and", &ast::bit_and, "");
    m.def("bit_or", &ast::bit_or, "");
    m.def("bit_xor", &ast::bit_xor, "");
    m.def("bit_shl", &ast::bit_shl, "");
    m.def("bit_shr", &ast::bit_shr, "");
    m.def("eq", &ast::eq, "");
    m.def("neq", &ast::neq, "");
    m.def("gt", &ast::gt, "");
    m.def("lt", &ast::lt, "");
    m.def("lte", &ast::lte, "");
    m.def("gte", &ast::gte, "");
    m.def("alloc_tensor", &ast::alloc_tensor, "");
    m.def("dim", &ast::dim, "");
}