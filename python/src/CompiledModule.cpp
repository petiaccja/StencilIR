#include "CompiledModule.hpp"

#include "Invoke.hpp"

#include <IR/ConvertOps.hpp>

#include <memory_resource>
#include <new>
#include <span>


namespace sir {


CompiledModule::CompiledModule(ops::ModuleOp ir, CompileOptions options)
    : m_ir(ir), m_options(options), m_functions(ExtractFunctions(ir)) {
}


pybind11::object CompiledModule::Invoke(std::string function, pybind11::args arguments) {
    if (!m_runner) {
        Compile();
    }

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

    ArgumentPack inputs{ functionType.parameters, m_runner.get() };
    ArgumentPack outputs{ functionType.returns, m_runner.get() };


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

    m_runner->Invoke(function, std::span{ opaquePointers });

    if (functionType.returns.size() > 1) {
        return outputs.Read(outputBuffer.get());
    }
    else if (functionType.returns.size() == 1) {
        auto results = outputs.Read(outputBuffer.get()).cast<pybind11::tuple>();
        return pybind11::reinterpret_borrow<pybind11::object>(*results.begin());
    }
    return pybind11::none{};
}


std::vector<StageResult> CompiledModule::GetStageResults() const {
    return m_stageResults;
}


void CompiledModule::Compile(bool recordStages) {
    auto pipeline = [&] {
        switch (m_options.accelerator) {
            case eAccelerator::NONE: return TargetCPUPipeline(m_context, m_options.optimizationOptions);
            case eAccelerator::CUDA: return TargetCUDAPipeline(m_context, m_options.optimizationOptions);
            default: throw std::invalid_argument("accelerator not currently implemeneted");
        }
    }();
    const int optLevel = static_cast<int>(m_options.optimizationLevel);

    const auto mlirIr = mlir::dyn_cast<mlir::ModuleOp>(ConvertOperation(m_context, m_ir));
    Compiler compiler(std::move(pipeline));
    auto llvmIr = recordStages ? compiler.Run(mlirIr, m_stageResults) : compiler.Run(mlirIr);

    m_runner = std::make_unique<Runner>(llvmIr, optLevel);
}


std::string CompiledModule::GetLLVMIR() const {
    return m_runner->GetLLVMIR();
}


std::vector<char> CompiledModule::GetObjectFile() const {
    return m_runner->GetObjectFile();
}


auto CompiledModule::ExtractFunctions(ops::ModuleOp ir) -> std::unordered_map<std::string, FunctionType> {
    std::unordered_map<std::string, FunctionType> functions;
    for (const auto& op : ir.GetBody().GetOperations()) {
        if (op.Type() == typeid(ops::FuncOp)) {
            const auto& attr = std::any_cast<const ops::FuncAttr&>(op.GetAttributes());
            functions.emplace(attr.name, FunctionType{ attr.signature->parameters, attr.signature->results });
        }
    }
    return functions;
}


} // namespace sir