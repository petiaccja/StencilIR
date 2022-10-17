#include "CompiledModule.hpp"

#include <memory_resource>


static Runner Compile(std::shared_ptr<ast::Module> ast,
                      CompileOptions options,
                      std::vector<StageResult>* stageResults = nullptr);
static auto ExtractFunctions(std::shared_ptr<ast::Module> ast)
    -> std::unordered_map<std::string, std::vector<ast::Type>>;
static void PythonToOpaque(pybind11::handle arg,
                           ast::Type type,
                           std::pmr::memory_resource& compatHeap,
                           std::pmr::vector<void*>& opaqueArgs);

thread_local std::vector<StageResult> stageResults; // TODO: fix this hack.

CompiledModule::CompiledModule(std::shared_ptr<ast::Module> ast, CompileOptions options, bool storeIr)
    : m_runner(Compile(ast, options, (stageResults = {}, storeIr ? &stageResults : nullptr))),
      m_functions(ExtractFunctions(ast)) {
    m_ir = stageResults;
}


void CompiledModule::Invoke(std::string function, pybind11::args args) {
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


std::vector<StageResult> CompiledModule::GetIR() const {
    return m_ir;
}


static Runner Compile(std::shared_ptr<ast::Module> ast, CompileOptions options, std::vector<StageResult>* stageResults) {
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
    auto llvm = stageResults ? compiler.Run(ir, *stageResults) : compiler.Run(ir);

    return Runner{ llvm, optLevel };
}


static auto ExtractFunctions(std::shared_ptr<ast::Module> ast)
    -> std::unordered_map<std::string, std::vector<ast::Type>> {
    std::unordered_map<std::string, std::vector<ast::Type>> functions;
    for (const auto& function : ast->functions) {
        std::vector<ast::Type> parameters;
        for (auto& parameter : function->parameters) {
            parameters.push_back(parameter.type);
        }
        functions.emplace(function->name, parameters);
    }
    return functions;
}

static std::string_view GetPythonFormatString(ast::ScalarType type) {
    return ast::VisitType(type, [](auto* t) {
        return pybind11::format_descriptor<std::decay_t<std::remove_pointer_t<decltype(t)>>>::value;
    });
}

static void PythonToOpaque(pybind11::handle arg,
                           ast::Type type,
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
    static const auto visitor = [&](auto type) {
        if constexpr (std::is_same_v<decltype(type), ast::ScalarType>) {
            ast::VisitType(type, [&](auto* t) {
                using T = std::decay_t<std::remove_pointer_t<decltype(t)>>;
                AppendArg(arg.cast<T>());
            });
        }
        else {
            const auto buffer = arg.cast<pybind11::buffer>();
            const auto request = buffer.request(true);
            if (request.format != GetPythonFormatString(type.elementType)) {
                throw std::invalid_argument("Buffer argument has different element type than the function expects.");
            }
            if (request.shape.size() != type.numDimensions) {
                throw std::invalid_argument("Buffer argument has different rank than the function expects.");
            }
            AppendArg(request.ptr); // ptr
            AppendArg(request.ptr); // aligned ptr
            AppendArg(size_t(0)); // offset
            for (auto& shape : request.shape) { // shape
                AppendArg(size_t(shape));
            }
            for (auto& stride : request.strides) { // strides
                if (stride % request.itemsize != 0) {
                    throw std::invalid_argument("Buffer strides must be multiples of itemsize.");
                }
                AppendArg(size_t(stride / request.itemsize));
            }
        }
    };
    std::visit(visitor, type);
}
