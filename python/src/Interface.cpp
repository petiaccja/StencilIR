#include "CompiledModule.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Diagnostics/Exception.hpp>
#include <IR/Ops.hpp>


template <class Range>
auto as_list(Range&& range) {
    pybind11::list list;
    for (const auto& v : range) {
        list.append(v);
    }
    return list;
}


void SubmoduleIR(pybind11::module_& main) {
    using namespace sir;
    using namespace ops;
    namespace py = pybind11;

    auto ops = main.def_submodule("ops");

    py::enum_<eArithmeticFunction>(ops, "ArithmeticFunction")
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


    py::enum_<eComparisonFunction>(ops, "ComparisonFunction")
        .value("EQ", eComparisonFunction::EQ)
        .value("NEQ", eComparisonFunction::NEQ)
        .value("LT", eComparisonFunction::LT)
        .value("GT", eComparisonFunction::GT)
        .value("LTE", eComparisonFunction::LTE)
        .value("GTE", eComparisonFunction::GTE)
        .export_values();

    py::class_<Location>(ops, "Location")
        .def(py::init<std::string, int, int>(), py::arg("file"), py::arg("line"), py::arg("column"))
        .def_readwrite("file", &Location::file)
        .def_readwrite("line", &Location::line)
        .def_readwrite("column", &Location::col);


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
        .def(py::init<std::string, std::shared_ptr<FunctionType>, bool, std::optional<Location>>(),
             py::arg("name"), py::arg("type"), py::arg("is_public"), py::arg("location"))
        .def("get_name", &FuncOp::GetName)
        .def("get_function_type", &FuncOp::GetFunctionType);
    py::class_<StencilOp, std::shared_ptr<StencilOp>>(ops, "StencilOp", singleRegion)
        .def(py::init<std::string, std::shared_ptr<FunctionType>, int, bool, std::optional<Location>>(),
             py::arg("name"), py::arg("type"), py::arg("num_dims"), py::arg("is_public"), py::arg("location"))
        .def("get_name", &StencilOp::GetName)
        .def("get_function_type", &StencilOp::GetFunctionType);
    py::class_<ReturnOp, std::shared_ptr<ReturnOp>>(ops, "ReturnOp", operation)
        .def(py::init<std::vector<Value>, std::optional<Location>>(),
             py::arg("values"), py::arg("location"))
        .def("get_values", &ReturnOp::GetValues);
    py::class_<CallOp, std::shared_ptr<CallOp>>(ops, "CallOp", operation)
        .def(py::init<FuncOp, std::vector<Value>, std::optional<Location>>(),
             py::arg("callee"), py::arg("args"), py::arg("location"))
        .def(py::init<std::string, std::vector<TypePtr>, std::vector<Value>, std::optional<Location>>(),
             py::arg("callee"), py::arg("results"), py::arg("args"), py::arg("location"))
        .def("get_callee", &CallOp::GetCallee)
        .def("get_num_results", &CallOp::GetNumResults)
        .def("get_args", &CallOp::GetArgs);
    py::class_<ApplyOp, std::shared_ptr<ApplyOp>>(ops, "ApplyOp", operation)
        .def(py::init<StencilOp, std::vector<Value>, std::vector<Value>, std::vector<Value>, std::vector<int64_t>, std::optional<Location>>(),
             py::arg("stencil"), py::arg("inputs"), py::arg("outputs"), py::arg("offsets"), py::arg("static_offsets"), py::arg("location"))
        .def(py::init<std::string, std::vector<Value>, std::vector<Value>, std::vector<Value>, std::vector<int64_t>, std::optional<Location>>(),
             py::arg("stencil"), py::arg("inputs"), py::arg("outputs"), py::arg("offsets"), py::arg("static_offsets"), py::arg("location"))
        .def("get_stencil", &ApplyOp::GetStencil)
        .def("get_num_results", &ApplyOp::GetNumResults)
        .def("get_inputs", &ApplyOp::GetInputs)
        .def("get_outputs", &ApplyOp::GetOutputs)
        .def("get_offsets", &ApplyOp::GetOffsets)
        .def("get_static_offsets", &ApplyOp::GetStaticOffsets);

    // Arithmetic-logic
    py::class_<CastOp, std::shared_ptr<CastOp>>(ops, "CastOp", operation)
        .def(py::init<Value, TypePtr, std::optional<Location>>(),
             py::arg("input"), py::arg("type"), py::arg("location"))
        .def("get_input", &CastOp::GetInput)
        .def("get_type", &CastOp::GetType)
        .def("get_result", &CastOp::GetResult);
    py::class_<ConstantOp, std::shared_ptr<ConstantOp>>(ops, "ConstantOp", operation)
        .def(py::init([](py::object value, TypePtr type, std::optional<Location> loc) {
                 if (py::isinstance(value, py::bool_().get_type())) {
                     return ConstantOp(py::cast<bool>(value), type, loc);
                 }
                 else if (py::isinstance(value, py::float_().get_type())) {
                     return ConstantOp(py::cast<double>(value), type, loc);
                 }
                 else if (py::isinstance(value, py::int_().get_type())) {
                     return ConstantOp(py::cast<int64_t>(value), type, loc);
                 }
                 throw std::invalid_argument("type of constant value is not understood");
             }),
             py::arg("value"), py::arg("type"), py::arg("location"))
        .def("get_value", &ConstantOp::GetValue)
        .def("get_type", &ConstantOp::GetType)
        .def("get_result", &ConstantOp::GetResult);
    py::class_<ArithmeticOp, std::shared_ptr<ArithmeticOp>>(ops, "ArithmeticOp", operation)
        .def(py::init<Value, Value, eArithmeticFunction, std::optional<Location>>(),
             py::arg("left"), py::arg("right"), py::arg("function"), py::arg("location"))
        .def("get_left", &ArithmeticOp::GetLeft)
        .def("get_right", &ArithmeticOp::GetRight)
        .def("get_result", &ArithmeticOp::GetResult)
        .def("get_function", &ArithmeticOp::GetFunction);
    py::class_<ComparisonOp, std::shared_ptr<ComparisonOp>>(ops, "ComparisonOp", operation)
        .def(py::init<Value, Value, eComparisonFunction, std::optional<Location>>(),
             py::arg("left"), py::arg("right"), py::arg("function"), py::arg("location"))
        .def("get_left", &ComparisonOp::GetLeft)
        .def("get_right", &ComparisonOp::GetRight)
        .def("get_result", &ComparisonOp::GetResult)
        .def("get_function", &ComparisonOp::GetFunction);
    py::class_<MinOp, std::shared_ptr<MinOp>>(ops, "MinOp", operation)
        .def(py::init<Value, Value, std::optional<Location>>(),
             py::arg("left"), py::arg("right"), py::arg("location"))
        .def("get_left", &MinOp::GetLeft)
        .def("get_right", &MinOp::GetRight)
        .def("get_result", &MinOp::GetResult);
    py::class_<MaxOp, std::shared_ptr<MaxOp>>(ops, "MaxOp", operation)
        .def(py::init<Value, Value, std::optional<Location>>(),
             py::arg("left"), py::arg("right"), py::arg("location"))
        .def("get_left", &MaxOp::GetLeft)
        .def("get_right", &MaxOp::GetRight)
        .def("get_result", &MaxOp::GetResult);

    // Control flow
    py::class_<IfOp, std::shared_ptr<IfOp>>(ops, "IfOp", operation)
        .def(py::init<Value, size_t, std::optional<Location>>(),
             py::arg("condition"), py::arg("num_results"), py::arg("location"))
        .def("get_condition", &IfOp::GetCondition)
        .def("get_then_region", [](IfOp& self) { return self.GetThenRegion(); })
        .def("get_else_region", [](IfOp& self) { return self.GetElseRegion(); });
    py::class_<ForOp, std::shared_ptr<ForOp>>(ops, "ForOp", singleRegion)
        .def(py::init<Value, Value, Value, std::vector<Value>, std::optional<Location>>(),
             py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("init"), py::arg("location"))
        .def("get_start", &ForOp::GetStart)
        .def("get_stop", &ForOp::GetStop)
        .def("get_step", &ForOp::GetStep);
    py::class_<YieldOp, std::shared_ptr<YieldOp>>(ops, "YieldOp", operation)
        .def(py::init<std::vector<Value>, std::optional<Location>>(),
             py::arg("values"), py::arg("location"))
        .def("get_values", &YieldOp::GetValues);

    // Tensor
    py::class_<DimOp, std::shared_ptr<DimOp>>(ops, "DimOp", operation)
        .def(py::init<Value, Value, std::optional<Location>>(),
             py::arg("source"), py::arg("index"), py::arg("location"))
        .def("get_result", &DimOp::GetResult);
    py::class_<AllocTensorOp, std::shared_ptr<AllocTensorOp>>(ops, "AllocTensorOp", operation)
        .def(py::init<TypePtr, std::vector<Value>, std::optional<Location>>(),
             py::arg("element_type"), py::arg("sizes"), py::arg("location"))
        .def("get_result", &AllocTensorOp::GetResult);
    py::class_<ExtractSliceOp, std::shared_ptr<ExtractSliceOp>>(ops, "ExtractSliceOp", operation)
        .def(py::init<Value, std::vector<Value>, std::vector<Value>, std::vector<Value>, std::optional<Location>>(),
             py::arg("source"), py::arg("offsets"), py::arg("sizes"), py::arg("strides"), py::arg("location"))
        .def("get_result", &ExtractSliceOp::GetResult);
    py::class_<InsertSliceOp, std::shared_ptr<InsertSliceOp>>(ops, "InsertSliceOp", operation)
        .def(py::init<Value, Value, std::vector<Value>, std::vector<Value>, std::vector<Value>, std::optional<Location>>(),
             py::arg("source"), py::arg("dest"), py::arg("offsets"), py::arg("sizes"), py::arg("strides"), py::arg("location"))
        .def("get_result", &InsertSliceOp::GetResult);

    // Stencil
    py::class_<IndexOp, std::shared_ptr<IndexOp>>(ops, "IndexOp", operation)
        .def(py::init<std::optional<Location>>(),
             py::arg("location"))
        .def("get_result", &IndexOp::GetResult);
    py::class_<JumpOp, std::shared_ptr<JumpOp>>(ops, "JumpOp", operation)
        .def(py::init<Value, std::vector<int64_t>, std::optional<Location>>(),
             py::arg("source"), py::arg("offsets"), py::arg("location"))
        .def("get_result", &JumpOp::GetResult);
    py::class_<ProjectOp, std::shared_ptr<ProjectOp>>(ops, "ProjectOp", operation)
        .def(py::init<Value, std::vector<int64_t>, std::optional<Location>>(),
             py::arg("source"), py::arg("positions"), py::arg("location"))
        .def("get_result", &ProjectOp::GetResult);
    py::class_<ExtendOp, std::shared_ptr<ExtendOp>>(ops, "ExtendOp", operation)
        .def(py::init<Value, int64_t, Value, std::optional<Location>>(),
             py::arg("source"), py::arg("position"), py::arg("value"), py::arg("location"))
        .def("get_result", &ExtendOp::GetResult);
    py::class_<ExchangeOp, std::shared_ptr<ExchangeOp>>(ops, "ExchangeOp", operation)
        .def(py::init<Value, int64_t, Value, std::optional<Location>>(),
             py::arg("source"), py::arg("position"), py::arg("value"), py::arg("location"))
        .def("get_result", &ExchangeOp::GetResult);
    py::class_<ExtractOp, std::shared_ptr<ExtractOp>>(ops, "ExtractOp", operation)
        .def(py::init<Value, int64_t, std::optional<Location>>(),
             py::arg("source"), py::arg("position"), py::arg("location"))
        .def("get_result", &ExtractOp::GetResult);
    py::class_<SampleOp, std::shared_ptr<SampleOp>>(ops, "SampleOp", operation)
        .def(py::init<Value, Value, std::optional<Location>>(),
             py::arg("source"), py::arg("index"), py::arg("location"))
        .def("get_result", &SampleOp::GetResult);
}


PYBIND11_MODULE(stencilir, m) {
    using namespace sir;

    m.doc() = "Stencil IR Python bindings";

    SubmoduleIR(m);

    pybind11::class_<CompiledModule>(m, "CompiledModule")
        .def(pybind11::init<ops::ModuleOp, CompileOptions>(), pybind11::arg("ir"), pybind11::arg("options"))
        .def("compile", &CompiledModule::Compile, pybind11::arg("record_stages") = false)
        .def("invoke", &CompiledModule::Invoke)
        .def("get_stage_results", &CompiledModule::GetStageResults)
        .def("get_llvm_ir", &CompiledModule::GetLLVMIR)
        .def("get_object_file", [](const CompiledModule& self) {
            const auto buffer = self.GetObjectFile();
            return pybind11::bytearray(buffer.data(), buffer.size());
        });


    //----------------------------------
    // Types
    //----------------------------------
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

    pybind11::class_<NDIndexType, std::shared_ptr<NDIndexType>>(m, "NDIndexType", type)
        .def(pybind11::init<int>())
        .def_readonly("num_dimensions", &NDIndexType::numDimensions);

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
    pybind11::enum_<eAccelerator>(m, "Accelerator")
        .value("NONE", eAccelerator::NONE)
        .value("CUDA", eAccelerator::CUDA)
        .value("AMDGPU", eAccelerator::AMDGPU)
        .export_values();

    pybind11::enum_<eOptimizationLevel>(m, "OptimizationLevel")
        .value("O0", eOptimizationLevel::O0)
        .value("O1", eOptimizationLevel::O1)
        .value("O2", eOptimizationLevel::O2)
        .value("O3", eOptimizationLevel::O3)
        .export_values();

    pybind11::class_<OptimizationOptions>(m, "OptimizationOptions")
        .def(pybind11::init<bool, bool, bool, bool, bool>(),
             pybind11::arg("inline_functions") = false,
             pybind11::arg("fuse_extract_slice_ops") = false,
             pybind11::arg("fuse_apply_ops") = false,
             pybind11::arg("eliminate_alloc_buffers") = false,
             pybind11::arg("enable_runtime_verification") = false)
        .def_readwrite("inline_functions", &OptimizationOptions::inlineFunctions)
        .def_readwrite("fuse_extract_slice_ops", &OptimizationOptions::fuseExtractSliceOps)
        .def_readwrite("fuse_apply_ops", &OptimizationOptions::fuseApplyOps)
        .def_readwrite("eliminate_alloc_buffers", &OptimizationOptions::eliminateAllocBuffers)
        .def_readwrite("enable_runtime_verification", &OptimizationOptions::enableRuntimeVerification)
        .def("__hash__", [](const OptimizationOptions& self) { return std::hash<OptimizationOptions>{}(self); })
        .def(pybind11::self == pybind11::self);

    pybind11::class_<CompileOptions>(m, "CompileOptions")
        .def(pybind11::init<eAccelerator, eOptimizationLevel, OptimizationOptions>(),
             pybind11::arg("accelerator"),
             pybind11::arg("opt_level"),
             pybind11::arg("opt_options") = OptimizationOptions{})
        .def_readwrite("accelerator", &CompileOptions::accelerator)
        .def_readwrite("opt_level", &CompileOptions::optimizationLevel)
        .def_readwrite("opt_options", &CompileOptions::optimizationOptions)
        .def("__hash__", [](const CompileOptions& self) { return std::hash<CompileOptions>{}(self); })
        .def(pybind11::self == pybind11::self);


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