#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/ASTNodes.hpp>
#include <AST/ConvertASTToIR.hpp>
#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <new>
#include <unordered_map>
#include <vector>


using namespace ast;


enum class eTargetArch {
    X86,
    NVPTX,
    AMDGPU,
};

enum class eOptimizationLevel {
    O0 = 0,
    O1 = 1,
    O2 = 2,
    O3 = 3,
};

struct CompileOptions {
    eTargetArch targetArch;
    eOptimizationLevel optimizationLevel;
};


class CompiledModule {
public:
    CompiledModule(std::shared_ptr<Module> ast, CompileOptions options)
        : m_runner(Compile(ast, options)),
          m_functions(ExtractFunctions(ast)) {}

    void Invoke(std::string function, pybind11::args args) {
        alignas(long double) thread_local char compatBuffer[256];
        std::pmr::monotonic_buffer_resource compatHeap{ compatBuffer, sizeof(compatBuffer), std::pmr::new_delete_resource() };
        alignas(long double) thread_local char opaqueBuffer[256];
        std::pmr::monotonic_buffer_resource opaqueHeap{ opaqueBuffer, sizeof(opaqueBuffer), std::pmr::new_delete_resource() };

        std::pmr::vector<void*> opaqueArgs{ std::pmr::polymorphic_allocator{ &opaqueHeap } };

        const auto functionIt = m_functions.find(function);
        if (functionIt == m_functions.end()) {
            throw std::invalid_argument("No function named '" + function + "' in compiled module.");
        }
        if (functionIt->second.size() != args.size()) {
            throw std::invalid_argument("Function '" + function + "' expects "
                                        + std::to_string(functionIt->second.size())
                                        + " arguments, " + std::to_string(args.size()) + "provided.");
        }

        auto typeIt = functionIt->second.begin();
        for (const auto& arg : args) {
            PythonToOpaque(arg, *(typeIt++), compatHeap, opaqueArgs);
        }

        m_runner.Invoke(function, std::span{ opaqueArgs });
    }

private:
    static Runner Compile(std::shared_ptr<Module> ast, CompileOptions options) {
        mlir::MLIRContext context;

        auto targetStages = [&] {
            switch (options.targetArch) {
                case eTargetArch::X86: return TargetCPUPipeline(context);
                default: throw std::invalid_argument("Target architecture not supported yet.");
            }
        }();
        const int optLevel = static_cast<int>(options.optimizationLevel);

        const auto ir = ConvertASTToIR(context, *ast);
        Compiler compiler(std::move(targetStages));
        auto llvm = compiler.Run(ir);

        return Runner{ llvm, optLevel };
    }

    static auto ExtractFunctions(std::shared_ptr<Module> ast) -> std::unordered_map<std::string, std::vector<types::Type>> {
        std::unordered_map<std::string, std::vector<types::Type>> functions;
        for (const auto& function : ast->functions) {
            std::vector<types::Type> parameters;
            for (auto& parameter : function->parameters) {
                parameters.push_back(parameter.type);
            }
            functions.emplace(function->name, parameters);
        }
        return functions;
    }

    static void PythonToOpaque(pybind11::handle arg,
                               types::Type type,
                               std::pmr::memory_resource& compatHeap,
                               std::pmr::vector<void*>& opaqueArgs) {
        static const auto AppendArg = [&](auto arg) {
            const auto compatibleArg = Runner::MakeCompatibleArgument(arg);

            // Must move compatibleArg to heap
            static_assert(std::is_trivially_destructible_v<decltype(compatibleArg)>, "Could allow other types as well.");
            void* const memoryLocation = compatHeap.allocate(sizeof(compatibleArg), alignof(decltype(compatibleArg)));
            auto& heapArg = *static_cast<std::remove_const_t<decltype(compatibleArg)>*>(memoryLocation);
            heapArg = compatibleArg;

            // Make opaque pointers to the argument.
            const auto opaqueArg = Runner::MakeOpaqueArgument(heapArg); // This is a tuple of void*'s
            std::apply([&](auto... opaquePointers) { (..., opaqueArgs.push_back(opaquePointers)); }, opaqueArg); // Essentially tuple foreach
        };
        struct {
            void operator()(const types::FundamentalType& type) const {
                switch (type.type) {
                    case types::FundamentalType::SINT8: AppendArg(int8_t(arg.cast<int>())); break;
                    case types::FundamentalType::SINT16: AppendArg(int16_t(arg.cast<int>())); break;
                    case types::FundamentalType::SINT32: AppendArg(int32_t(arg.cast<int>())); break;
                    case types::FundamentalType::SINT64: AppendArg(int64_t(arg.cast<int>())); break;
                    case types::FundamentalType::UINT8: AppendArg(uint8_t(arg.cast<int>())); break;
                    case types::FundamentalType::UINT16: AppendArg(uint16_t(arg.cast<int>())); break;
                    case types::FundamentalType::UINT32: AppendArg(uint32_t(arg.cast<int>())); break;
                    case types::FundamentalType::UINT64: AppendArg(uint64_t(arg.cast<int>())); break;
                    case types::FundamentalType::SSIZE: AppendArg(ptrdiff_t(arg.cast<int>())); break;
                    case types::FundamentalType::USIZE: AppendArg(size_t(arg.cast<int>())); break;
                    case types::FundamentalType::FLOAT32: AppendArg(float(arg.cast<double>())); break;
                    case types::FundamentalType::FLOAT64: AppendArg(double(arg.cast<double>())); break;
                    case types::FundamentalType::BOOL: AppendArg(arg.cast<bool>()); break;
                    default: throw std::invalid_argument("Invalid type.");
                }
            }
            void operator()(const types::FieldType& type) const {
                const auto buffer = arg.cast<pybind11::buffer>();
                const auto request = buffer.request(true);
                // TODO: check for item type as well.
                if (request.shape.size() != type.numDimensions) {
                    throw std::invalid_argument("Field dimension mismatch.");
                }
                AppendArg(request.ptr); // ptr
                AppendArg(request.ptr); // aligned ptr
                AppendArg(size_t(0)); // offset
                for (auto& dim : request.shape) { // shape
                    AppendArg(size_t(dim));
                }
                for (auto& dim : request.strides) { // strides
                    AppendArg(size_t(dim));
                }
            }
            pybind11::handle arg;
            std::pmr::memory_resource& compatHeap;
            std::pmr::vector<void*>& opaqueArgs;
        } visitor{ arg, compatHeap, opaqueArgs };
    }

private:
    Runner m_runner;
    std::unordered_map<std::string, std::vector<types::Type>> m_functions;
};


CompiledModule Compile(std::shared_ptr<Module> ast, CompileOptions options) {
    return CompiledModule{ ast, options };
}


PYBIND11_MODULE(stencilir_bindings, m) {
    m.doc() = "Stencil IR Python bindings";

    //----------------------------------
    // AST nodes
    //----------------------------------
    pybind11::class_<Statement, std::shared_ptr<Statement>> statement(m, "Statement");

    pybind11::class_<Expression, std::shared_ptr<Expression>> expression(m, "Expression", statement);

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

    pybind11::class_<Return, std::shared_ptr<Return>>(m, "Return", statement)
        .def(pybind11::init<std::vector<std::shared_ptr<Expression>>,
                            std::optional<Location>>());

    //----------------------------------
    // AST structures
    //----------------------------------
    pybind11::class_<Parameter>(m, "Parameter")
        .def(pybind11::init<std::string,
                            types::Type>());

    pybind11::class_<Location>(m, "Location")
        .def(pybind11::init<std::string,
                            int,
                            int>());


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