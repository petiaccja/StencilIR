#include "CompiledModule.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/Nodes.hpp>
#include <DAG/Ops.hpp>
#include <Diagnostics/Exception.hpp>


template <class Range>
auto as_list(Range&& range) {
    pybind11::list list;
    for (const auto& v : range) {
        list.append(v);
    }
    return list;
    // return std::vector{ std::begin(range), std::end(range) };
}


void SubmoduleIR(pybind11::module_& main) {
    using namespace dag;
    namespace py = pybind11;

    auto ops = main.def_submodule("ops");

    py::enum_<dag::eArithmeticFunction>(ops, "ArithmeticFunction")
        .value("ADD", dag::eArithmeticFunction::ADD)
        .value("SUB", dag::eArithmeticFunction::SUB)
        .value("MUL", dag::eArithmeticFunction::MUL)
        .value("DIV", dag::eArithmeticFunction::DIV)
        .value("MOD", dag::eArithmeticFunction::MOD)
        .value("BIT_AND", dag::eArithmeticFunction::BIT_AND)
        .value("BIT_OR", dag::eArithmeticFunction::BIT_OR)
        .value("BIT_XOR", dag::eArithmeticFunction::BIT_XOR)
        .value("BIT_SHL", dag::eArithmeticFunction::BIT_SHL)
        .value("BIT_SHR", dag::eArithmeticFunction::BIT_SHR)
        .export_values();


    py::enum_<dag::eComparisonFunction>(ops, "ComparisonFunction")
        .value("EQ", dag::eComparisonFunction::EQ)
        .value("NEQ", dag::eComparisonFunction::NEQ)
        .value("LT", dag::eComparisonFunction::LT)
        .value("GT", dag::eComparisonFunction::GT)
        .value("LTE", dag::eComparisonFunction::LTE)
        .value("GTE", dag::eComparisonFunction::GTE)
        .export_values();

    py::class_<dag::Location>(ops, "Location")
        .def(py::init<std::string, int, int>(), py::arg("file"), py::arg("line"), py::arg("column"))
        .def_readwrite("file", &dag::Location::file)
        .def_readwrite("line", &dag::Location::line)
        .def_readwrite("column", &dag::Location::col);


    // Base classes
    py::class_<Value, std::shared_ptr<Value>>(ops, "Value")
        .def("owner", &Value::GetOwner)
        .def("index", &Value::GetIndex);
    py::class_<Operand, std::shared_ptr<Operand>>(ops, "Operand")
        .def("owner", &Operand::GetOwner)
        .def("source", &Operand::GetSource);
    py::class_<Region, std::shared_ptr<Region>>(ops, "Region")
        .def("add", [](Region& self, py::object op) {
            auto asOp = py::cast<Operation>(op);
            self.GetOperations().push_back(asOp);
            return op;
        })
        .def("get_operations", [](Region& self) { return self.GetOperations(); })
        .def("get_args", [](Region& self) { return self.GetArgs(); });
    py::class_<Operation, std::shared_ptr<Operation>> operation(ops, "Operation");
    operation.def("get_operands", [](Operation& self) { return as_list(self.GetOperands()); })
        .def("get_results", [](Operation& self) { return as_list(self.GetResults()); })
        .def("get_regions", [](Operation& self) { return as_list(self.GetRegions()); })
        .def("get_location", &Operation::GetLocation);
    py::class_<SingleRegion, std::shared_ptr<SingleRegion>> singleRegion(ops, "SingleRegion", operation);
    singleRegion.def("get_body", [](SingleRegion& self) { return self.GetBody(); })
        .def("get_num_region_args", &SingleRegion::GetNumRegionArgs)
        .def("get_region_args", &SingleRegion::GetRegionArgs)
        .def("get_region_arg", &SingleRegion::GetRegionArg)
        .def("add", [](SingleRegion& self, py::object op) {
            auto asOp = py::cast<Operation>(op);
            self.GetBody().GetOperations().push_back(asOp);
            return op;
        });

    // Module structure
    py::class_<ModuleOp, std::shared_ptr<ModuleOp>>(ops, "ModuleOp", singleRegion)
        .def(py::init<>());
    py::class_<FuncOp, std::shared_ptr<FuncOp>>(ops, "FuncOp", singleRegion)
        .def(py::init<std::string, std::shared_ptr<ast::FunctionType>, bool, std::optional<dag::Location>>(),
             py::arg("name"), py::arg("type"), py::arg("is_public"), py::arg("location"))
        .def("get_name", &FuncOp::GetName)
        .def("get_function_type", &FuncOp::GetFunctionType);
    py::class_<StencilOp, std::shared_ptr<StencilOp>>(ops, "StencilOp", singleRegion)
        .def(py::init<std::string, std::shared_ptr<ast::FunctionType>, int, bool, std::optional<dag::Location>>(),
             py::arg("name"), py::arg("type"), py::arg("num_dims"), py::arg("is_public"), py::arg("location"))
        .def("get_name", &StencilOp::GetName)
        .def("get_function_type", &StencilOp::GetFunctionType);
    py::class_<ReturnOp, std::shared_ptr<ReturnOp>>(ops, "ReturnOp", operation)
        .def(py::init<std::vector<Value>, std::optional<dag::Location>>(),
             py::arg("values"), py::arg("location"))
        .def("get_values", &ReturnOp::GetValues);
    py::class_<CallOp, std::shared_ptr<CallOp>>(ops, "CallOp", operation)
        .def(py::init<FuncOp, std::vector<Value>, std::optional<dag::Location>>(),
             py::arg("callee"), py::arg("args"), py::arg("location"))
        .def("get_callee", &CallOp::GetCallee)
        .def("get_num_results", &CallOp::GetNumResults)
        .def("get_args", &CallOp::GetArgs);
    py::class_<ApplyOp, std::shared_ptr<ApplyOp>>(ops, "ApplyOp", operation)
        .def(py::init<StencilOp, std::vector<Value>, std::vector<Value>, std::vector<Value>, std::vector<int64_t>, std::optional<dag::Location>>(),
             py::arg("stencil"), py::arg("inputs"), py::arg("outputs"), py::arg("offsets"), py::arg("static_offsets"), py::arg("location"))
        .def("get_stencil", &ApplyOp::GetStencil)
        .def("get_num_results", &ApplyOp::GetNumResults)
        .def("get_inputs", &ApplyOp::GetInputs)
        .def("get_outputs", &ApplyOp::GetOutputs)
        .def("get_offsets", &ApplyOp::GetOffsets)
        .def("get_static_offsets", &ApplyOp::GetStaticOffsets);

    // Arithmetic-logic
    py::class_<CastOp, std::shared_ptr<CastOp>>(ops, "CastOp", operation)
        .def(py::init<Value, ast::TypePtr, std::optional<dag::Location>>(),
             py::arg("input"), py::arg("type"), py::arg("location"))
        .def("get_input", &CastOp::GetInput)
        .def("get_type", &CastOp::GetType)
        .def("get_result", &CastOp::GetResult);
    py::class_<ConstantOp, std::shared_ptr<ConstantOp>>(ops, "ConstantOp", operation)
        .def(py::init([](py::object value, ast::TypePtr type, std::optional<dag::Location> loc) {
                 try {
                     return ConstantOp(py::cast<bool>(value), type, loc);
                 }
                 catch (...) {
                 }
                 try {
                     return ConstantOp(py::cast<double>(value), type, loc);
                 }
                 catch (...) {
                 }
                 try {
                     return ConstantOp(py::cast<int64_t>(value), type, loc);
                 }
                 catch (...) {
                 }
                 throw std::invalid_argument("type of constant value is not understood");
             }),
             py::arg("value"), py::arg("type"), py::arg("location"))
        .def("get_value", &ConstantOp::GetValue)
        .def("get_type", &ConstantOp::GetType)
        .def("get_result", &ConstantOp::GetResult);
    py::class_<ArithmeticOp, std::shared_ptr<ArithmeticOp>>(ops, "ArithmeticOp", operation)
        .def(py::init<Value, Value, eArithmeticFunction, std::optional<dag::Location>>(),
             py::arg("left"), py::arg("right"), py::arg("function"), py::arg("location"))
        .def("get_left", &ArithmeticOp::GetLeft)
        .def("get_right", &ArithmeticOp::GetRight)
        .def("get_result", &ArithmeticOp::GetResult)
        .def("get_function", &ArithmeticOp::GetFunction);
    py::class_<ComparisonOp, std::shared_ptr<ComparisonOp>>(ops, "ComparisonOp", operation)
        .def(py::init<Value, Value, eComparisonFunction, std::optional<dag::Location>>(),
             py::arg("left"), py::arg("right"), py::arg("function"), py::arg("location"))
        .def("get_left", &ComparisonOp::GetLeft)
        .def("get_right", &ComparisonOp::GetRight)
        .def("get_result", &ComparisonOp::GetResult)
        .def("get_function", &ComparisonOp::GetFunction);
    py::class_<MinOp, std::shared_ptr<MinOp>>(ops, "MinOp", operation)
        .def(py::init<Value, Value, std::optional<dag::Location>>(),
             py::arg("left"), py::arg("right"), py::arg("location"))
        .def("get_left", &MinOp::GetLeft)
        .def("get_right", &MinOp::GetRight)
        .def("get_result", &MinOp::GetResult);
    py::class_<MaxOp, std::shared_ptr<MaxOp>>(ops, "MaxOp", operation)
        .def(py::init<Value, Value, std::optional<dag::Location>>(),
             py::arg("left"), py::arg("right"), py::arg("location"))
        .def("get_left", &MaxOp::GetLeft)
        .def("get_right", &MaxOp::GetRight)
        .def("get_result", &MaxOp::GetResult);

    // Control flow
    py::class_<IfOp, std::shared_ptr<IfOp>>(ops, "IfOp", operation)
        .def(py::init<Value, size_t, std::optional<dag::Location>>(),
             py::arg("condition"), py::arg("num_results"), py::arg("location"))
        .def("get_condition", &IfOp::GetCondition)
        .def("get_then_region", [](IfOp& self) { return self.GetThenRegion(); })
        .def("get_else_region", [](IfOp& self) { return self.GetElseRegion(); });
    py::class_<ForOp, std::shared_ptr<ForOp>>(ops, "ForOp", singleRegion)
        .def(py::init<Value, Value, Value, std::vector<Value>, std::optional<dag::Location>>(),
             py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("init"), py::arg("location"))
        .def("get_start", &ForOp::GetStart)
        .def("get_stop", &ForOp::GetStop)
        .def("get_step", &ForOp::GetStep);
    py::class_<YieldOp, std::shared_ptr<YieldOp>>(ops, "YieldOp", operation)
        .def(py::init<std::vector<Value>, std::optional<dag::Location>>(),
             py::arg("values"), py::arg("location"))
        .def("get_values", &YieldOp::GetValues);

    // Tensor
    py::class_<DimOp, std::shared_ptr<DimOp>>(ops, "DimOp", operation)
        .def(py::init<Value, Value, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("index"), py::arg("location"))
        .def("get_result", &DimOp::GetResult);
    py::class_<AllocTensorOp, std::shared_ptr<AllocTensorOp>>(ops, "AllocTensorOp", operation)
        .def(py::init<ast::TypePtr, std::vector<Value>, std::optional<dag::Location>>(),
             py::arg("element_type"), py::arg("sizes"), py::arg("location"))
        .def("get_result", &AllocTensorOp::GetResult);
    py::class_<ExtractSliceOp, std::shared_ptr<ExtractSliceOp>>(ops, "ExtractSliceOp", operation)
        .def(py::init<Value, std::vector<Value>, std::vector<Value>, std::vector<Value>, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("offsets"), py::arg("sizes"), py::arg("strides"), py::arg("location"))
        .def("get_result", &ExtractSliceOp::GetResult);
    py::class_<InsertSliceOp, std::shared_ptr<InsertSliceOp>>(ops, "InsertSliceOp", operation)
        .def(py::init<Value, Value, std::vector<Value>, std::vector<Value>, std::vector<Value>, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("dest"), py::arg("offsets"), py::arg("sizes"), py::arg("strides"), py::arg("location"))
        .def("get_result", &InsertSliceOp::GetResult);

    // Stencil
    py::class_<IndexOp, std::shared_ptr<IndexOp>>(ops, "IndexOp", operation)
        .def(py::init<std::optional<dag::Location>>(),
             py::arg("location"))
        .def("get_result", &IndexOp::GetResult);
    py::class_<JumpOp, std::shared_ptr<JumpOp>>(ops, "JumpOp", operation)
        .def(py::init<Value, std::vector<int64_t>, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("offsets"), py::arg("location"))
        .def("get_result", &JumpOp::GetResult);
    py::class_<ProjectOp, std::shared_ptr<ProjectOp>>(ops, "ProjectOp", operation)
        .def(py::init<Value, std::vector<int64_t>, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("positions"), py::arg("location"))
        .def("get_result", &ProjectOp::GetResult);
    py::class_<ExtendOp, std::shared_ptr<ExtendOp>>(ops, "ExtendOp", operation)
        .def(py::init<Value, int64_t, Value, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("position"), py::arg("value"), py::arg("location"))
        .def("get_result", &ExtendOp::GetResult);
    py::class_<ExchangeOp, std::shared_ptr<ExchangeOp>>(ops, "ExchangeOp", operation)
        .def(py::init<Value, int64_t, Value, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("position"), py::arg("value"), py::arg("location"))
        .def("get_result", &ExchangeOp::GetResult);
    py::class_<ExtractOp, std::shared_ptr<ExtractOp>>(ops, "ExtractOp", operation)
        .def(py::init<Value, int64_t, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("position"), py::arg("location"))
        .def("get_result", &ExtractOp::GetResult);
    py::class_<SampleOp, std::shared_ptr<SampleOp>>(ops, "SampleOp", operation)
        .def(py::init<Value, Value, std::optional<dag::Location>>(),
             py::arg("source"), py::arg("index"), py::arg("location"))
        .def("get_result", &SampleOp::GetResult);
}


using namespace ast;


PYBIND11_MODULE(stencilir, m) {
    m.doc() = "Stencil IR Python bindings";

    SubmoduleIR(m);

    pybind11::class_<CompiledModule>(m, "CompiledModule")
        .def(pybind11::init<std::shared_ptr<ast::Module>, CompileOptions>(), pybind11::arg("ast"), pybind11::arg("options"))
        .def(pybind11::init<dag::ModuleOp, CompileOptions>(), pybind11::arg("ir"), pybind11::arg("options"))
        .def("compile", &CompiledModule::Compile)
        .def("invoke", &CompiledModule::Invoke)
        .def("get_stage_results", &CompiledModule::GetStageResults);

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

    pybind11::class_<FunctionType, std::shared_ptr<FunctionType>>(m, "FunctionType", type)
        .def(pybind11::init<std::vector<TypePtr>, std::vector<TypePtr>>())
        .def_readonly("parameters", &FunctionType::parameters)
        .def_readonly("results", &FunctionType::results);


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


    pybind11::class_<StageResult>(m, "StageIR")
        .def_readonly("stage", &StageResult::name)
        .def_readonly("ir", &StageResult::ir);

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
            compilationError(ss.str().c_str());
        }
    });
}