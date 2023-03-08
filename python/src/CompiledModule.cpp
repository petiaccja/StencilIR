#include "CompiledModule.hpp"

#include "Invoke.hpp"

#include <memory_resource>
#include <new>
#include <span>


static thread_local std::vector<StageResult> stageResults; // TODO: fix this hack.


CompiledModule::CompiledModule(std::shared_ptr<ast::Module> ast, CompileOptions options, bool storeIr)
    : m_runner(Compile(ast, options, (stageResults = {}, storeIr ? &stageResults : nullptr))),
      m_functions(ExtractFunctions(ast)) {
    m_ir = stageResults;
}


pybind11::object CompiledModule::Invoke(std::string function, pybind11::args arguments) {
    const auto& functionIt = m_functions.find(function);
    if (functionIt == m_functions.end()) {
        throw std::invalid_argument("no function named '" + function + "' in compiled module");
    }

    const auto& functionType = functionIt->second;
    if (functionType.parameters.size() != arguments.size()) {
        throw std::invalid_argument("function '" + function + "' expects "
                                    + std::to_string(functionType.parameters.size())
                                    + " arguments, " + std::to_string(arguments.size()) + " provided");
    }

    ArgumentPack inputs{ functionType.parameters, &m_runner };
    ArgumentPack outputs{ functionType.returns, &m_runner };


    const auto inputAlignment = std::align_val_t{ inputs.GetAlignment() };
    const auto outputAlignment = std::align_val_t{ outputs.GetAlignment() };
    struct AlignedDeleter {
        std::align_val_t alignment;
        void operator()(void* ptr) const noexcept { operator delete(ptr, alignment); }
    };
    auto inputBuffer = std::unique_ptr<std::byte[], AlignedDeleter>(static_cast<std::byte*>(operator new(sizeof(std::byte) * inputs.GetSize(), inputAlignment)), { inputAlignment });
    auto outputBuffer = std::unique_ptr<std::byte[], AlignedDeleter>(static_cast<std::byte*>(operator new(sizeof(std::byte) * inputs.GetSize(), outputAlignment)), { outputAlignment });

    inputs.Write(arguments, inputBuffer.get());
    std::vector<void*> opaquePointers;
    opaquePointers.reserve(arguments.size() + 1);
    inputs.GetOpaquePointers(inputBuffer.get(), std::back_inserter(opaquePointers));
    opaquePointers.push_back(outputBuffer.get());

    m_runner.Invoke(function, std::span{ opaquePointers });

    if (functionType.returns.size() > 1) {
        return outputs.Read(outputBuffer.get());
    }
    else if (functionType.returns.size() == 1) {
        auto results = outputs.Read(outputBuffer.get()).cast<pybind11::tuple>();
        return pybind11::reinterpret_borrow<pybind11::object>(*results.begin());
    }
    return pybind11::none{};
}


std::vector<StageResult> CompiledModule::GetIR() const {
    return m_ir;
}


Runner CompiledModule::Compile(std::shared_ptr<ast::Module> ast, CompileOptions options, std::vector<StageResult>* stageResults) {
    mlir::MLIRContext context;

    auto targetStages = [&] {
        switch (options.targetArch) {
            case eTargetArch::X86: return TargetCPUPipeline(context, options.optimizationOptions);
            default: throw std::invalid_argument("Target architecture not supported yet.");
        }
    }();
    const int optLevel = static_cast<int>(options.optimizationLevel);

    const auto ir = ConvertASTToIR(context, *ast);
    Compiler compiler(std::move(targetStages));
    auto llvm = stageResults ? compiler.Run(ir, *stageResults) : compiler.Run(ir);

    return Runner{ llvm, optLevel };
}


auto CompiledModule::ExtractFunctions(std::shared_ptr<ast::Module> ast)
    -> std::unordered_map<std::string, FunctionType> {
    std::unordered_map<std::string, FunctionType> functions;
    for (const auto& function : ast->functions) {
        std::vector<ast::TypePtr> parameters;
        for (auto& parameter : function->parameters) {
            parameters.push_back(parameter.type);
        }
        functions.emplace(function->name, FunctionType{ parameters, function->results });
    }
    return functions;
}