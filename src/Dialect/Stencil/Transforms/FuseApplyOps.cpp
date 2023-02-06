#include "FuseApplyOps.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <ranges>
#include <regex>
#include <span>


using namespace mlir;


bool IsApplyOpInputFusable(stencil::ApplyOp applyOp, size_t operandIdx) {
    const mlir::Value input = applyOp.getInputs()[operandIdx];
    const auto definingOp = input.getDefiningOp();
    const bool isDefinedByApply = definingOp && mlir::isa<stencil::ApplyOp>(definingOp);

    const auto stencil = mlir::SymbolTable::lookupNearestSymbolFrom<stencil::StencilOp>(applyOp, applyOp.getCalleeAttr());
    const mlir::Value blockArg = stencil->getRegion(0).getBlocks().begin()->getArgument(operandIdx);
    const auto blockArgUsers = blockArg.getUsers();
    const bool isOnlyUsedBySample = std::all_of(blockArgUsers.begin(), blockArgUsers.end(), [](auto&& user) {
        return mlir::isa<stencil::SampleOp>(user);
    });

    return isDefinedByApply && isOnlyUsedBySample;
}


struct MemoryAccessPattern {
    int numMemoryAccesses;
};


MemoryAccessPattern AnalyzeMemoryAccesses(mlir::Region& region) {
    MemoryAccessPattern pattern{ .numMemoryAccesses = 0 };
    region.walk([&](mlir::Operation* op) {
        MemoryAccessPattern opPattern{ .numMemoryAccesses = 0 };
        if (auto branchOp = mlir::dyn_cast<mlir::RegionBranchOpInterface>(op)) {
            // Assume worst of all regions.
            for (auto& branchRegion : branchOp->getRegions()) {
                const auto branchPattern = AnalyzeMemoryAccesses(branchRegion);
                opPattern.numMemoryAccesses = std::max(opPattern.numMemoryAccesses, branchPattern.numMemoryAccesses);
            }
        }
        else if (auto loopOp = mlir::dyn_cast<mlir::LoopLikeOpInterface>(op)) {
            // Assume large loop.
            // TODO: consider fixed-size loops and hints.
            opPattern.numMemoryAccesses = 10000;
        }
        else if (auto sampleOp = mlir::dyn_cast<stencil::SampleOp>(op)) {
            opPattern.numMemoryAccesses = 1;
        }
        pattern.numMemoryAccesses += opPattern.numMemoryAccesses;
        return mlir::WalkResult::advance();
    });
    return pattern;
}


float FusePrecedingStencilCost(stencil::StencilOp precedingStencil,
                               stencil::StencilOp targetStencil,
                               std::span<const std::pair<size_t, size_t>> resultsToParams) {
    constexpr float writeCost = 1.5f;
    constexpr float readCost = 1.0f;

    const auto memoryAccessPattern = AnalyzeMemoryAccesses(precedingStencil.getRegion());
    size_t numUsesTotal = 0;
    for (const auto& [resultIdx, paramIdx] : resultsToParams) {
        const auto uses = targetStencil.getRegion().getBlocks().front().getArgument(paramIdx).getUses();
        const size_t numUses = std::distance(uses.begin(), uses.end());
        numUsesTotal += numUses;
    }

    const float originalCost = writeCost + memoryAccessPattern.numMemoryAccesses * readCost;
    const float fusedCost = memoryAccessPattern.numMemoryAccesses * numUsesTotal * readCost;
    return fusedCost - originalCost;
}


mlir::StringAttr UniqueFusedStencilName(stencil::StencilOp originalStencil, mlir::PatternRewriter& rewriter) {
    std::string symName = originalStencil.getSymName().str();
    std::regex format{ R"((.*)(_fused_)([0-9]+))" };
    std::smatch match;
    std::regex_match(symName, match, format);
    if (!match.empty()) {
        symName = match[0];
    }

    size_t serial = 1;
    mlir::StringAttr fusedSymName;
    do {
        auto newName = symName + "_fused_" + std::to_string(serial);
        fusedSymName = rewriter.getStringAttr(std::move(newName));
        ++serial;
    } while (nullptr != mlir::SymbolTable::lookupNearestSymbolFrom(originalStencil, fusedSymName));
    return fusedSymName;
}


auto FusePrecedingStencilOp(stencil::StencilOp precedingStencil,
                            stencil::StencilOp targetStencil,
                            std::span<const std::pair<size_t, size_t>> resultsToParams,
                            mlir::PatternRewriter& rewriter) -> mlir::FailureOr<stencil::StencilOp> {
    rewriter.setInsertionPointAfter(targetStencil);
    stencil::StencilOp fusedStencil = mlir::dyn_cast<stencil::StencilOp>(rewriter.clone(*targetStencil));
    auto& sourceEntryBlock = precedingStencil.getRegion().front();
    auto& fusedEntryBlock = fusedStencil.getRegion().front();
    mlir::SmallVector<mlir::Type, 16> fusedParamTypes{ targetStencil.getFunctionType().getInputs().begin(),
                                                       targetStencil.getFunctionType().getInputs().end() };
    const auto fusedResultTypes = fusedStencil.getFunctionType().getResults();

    // Append source block arguments to fused.
    for (auto& sourceBlockArg : sourceEntryBlock.getArguments()) {
        fusedEntryBlock.addArgument(sourceBlockArg.getType(), sourceBlockArg.getLoc());
        fusedParamTypes.push_back(sourceBlockArg.getType());
    }

    // Replace sample ops with invoke ops.
    mlir::SmallVector<mlir::Value, 16> invokeArgs = {
        fusedEntryBlock.args_end() - sourceEntryBlock.getArguments().size(),
        fusedEntryBlock.args_end()
    };
    mlir::InlinerInterface inlinerInterface{ rewriter.getContext() };
    for (const auto& [resultIdx, paramIdx] : resultsToParams) {
        auto blockArg = fusedEntryBlock.getArgument(paramIdx);
        for (auto user : blockArg.getUsers()) {
            auto sampleOp = mlir::dyn_cast<stencil::SampleOp>(user);
            if (!sampleOp) {
                rewriter.eraseOp(fusedStencil);
                return mlir::failure();
            }
            rewriter.setInsertionPointAfter(sampleOp);
            auto invokeOp = rewriter.create<stencil::InvokeOp>(sampleOp->getLoc(),
                                                               precedingStencil.getFunctionType().getResults(),
                                                               precedingStencil.getSymName(),
                                                               sampleOp.getIndex(),
                                                               invokeArgs);
            rewriter.replaceOp(sampleOp, { invokeOp->getResult(resultIdx) });
            if (failed(mlir::inlineCall(inlinerInterface, std::move(invokeOp), precedingStencil, &precedingStencil.getRegion()))) {
                rewriter.eraseOp(fusedStencil);
                return mlir::failure();
            }
            rewriter.eraseOp(invokeOp);
        }
    }

    // Erase unused block args
    std::vector<size_t> paramsToRemove;
    for (const auto& [resultIdx, paramIdx] : resultsToParams) {
        paramsToRemove.push_back(paramIdx);
    }
    std::ranges::sort(paramsToRemove);
    std::ranges::reverse(paramsToRemove);
    for (auto paramIdx : paramsToRemove) {
        fusedEntryBlock.eraseArgument(paramIdx);
        fusedParamTypes.erase(fusedParamTypes.begin() + paramIdx);
    }

    // Update function type
    fusedStencil.setFunctionTypeAttr(mlir::TypeAttr::get(rewriter.getFunctionType(fusedParamTypes, fusedResultTypes)));
    fusedStencil.setSymNameAttr(UniqueFusedStencilName(precedingStencil, rewriter));

    return fusedStencil;
}


auto FusePrecedingApplyOp(stencil::ApplyOp precedingOp,
                          stencil::ApplyOp targetOp,
                          std::span<const std::pair<size_t, size_t>> resultsToParams,
                          mlir::PatternRewriter& rewriter) -> mlir::FailureOr<stencil::ApplyOp> {
    const auto precedingStencil = mlir::SymbolTable::lookupNearestSymbolFrom<stencil::StencilOp>(precedingOp, precedingOp.getCalleeAttr());
    const auto targetStencil = mlir::SymbolTable::lookupNearestSymbolFrom<stencil::StencilOp>(targetOp, targetOp.getCalleeAttr());

    auto maybeFusedStencil = FusePrecedingStencilOp(precedingStencil, targetStencil, resultsToParams, rewriter);
    if (failed(maybeFusedStencil)) {
        return failure();
    }
    auto fusedStencil = maybeFusedStencil.value();

    // Create fused apply op
    mlir::SmallVector<mlir::Value, 16> inputs;
    std::copy(targetOp.getInputs().begin(), targetOp.getInputs().end(), std::back_inserter(inputs));
    std::copy(precedingOp.getInputs().begin(), precedingOp.getInputs().end(), std::back_inserter(inputs));

    std::vector<size_t> inputsToRemove;
    for (const auto& [resultIdx, paramIdx] : resultsToParams) {
        inputsToRemove.push_back(paramIdx);
    }
    std::ranges::sort(inputsToRemove);
    std::ranges::reverse(inputsToRemove);
    for (auto inputIdx : inputsToRemove) {
        inputs.erase(inputs.begin() + inputIdx);
    }

    rewriter.setInsertionPointAfter(targetOp);
    auto fusedApplyOp = rewriter.create<stencil::ApplyOp>(targetOp->getLoc(),
                                                          targetOp->getResultTypes(),
                                                          fusedStencil.getSymName(),
                                                          inputs,
                                                          targetOp.getOutputs(),
                                                          targetOp.getOffsets(),
                                                          targetOp.getStaticOffsetsAttr());

    return fusedApplyOp;
}

auto CollectPrecedingApplyOps(stencil::ApplyOp targetOp) -> mlir::SmallVector<stencil::ApplyOp, 8> {
    mlir::SmallVector<stencil::ApplyOp, 8> precedingApplyOps;
    for (auto input : targetOp.getInputs()) {
        if (const auto definingOp = input.getDefiningOp()) {
            if (auto definingApply = mlir::dyn_cast<stencil::ApplyOp>(definingOp)) {
                precedingApplyOps.push_back(definingApply);
            }
        }
    }
    std::sort(precedingApplyOps.begin(), precedingApplyOps.end());
    precedingApplyOps.erase(std::unique(precedingApplyOps.begin(), precedingApplyOps.end()), precedingApplyOps.end());
    return precedingApplyOps;
}


auto MapPrecedingOpResults(mlir::Operation* preceding, mlir::Operation* target)
    -> mlir::SmallVector<std::pair<size_t, size_t>, 4> {
    mlir::SmallVector<std::pair<size_t, size_t>, 4> map;
    for (auto result : preceding->getResults()) {
        for (auto& input : target->getOpOperands()) {
            if (input.is(result)) {
                map.emplace_back(result.getResultNumber(), input.getOperandNumber());
            }
        }
    }
    return map;
}


auto FusePrecedingApplies(stencil::ApplyOp targetOp, mlir::PatternRewriter& rewriter)
    -> mlir::FailureOr<std::pair<stencil::ApplyOp, mlir::SmallVector<stencil::ApplyOp, 12>>> {
    const auto primaryTargetOp = targetOp;
    const auto precedingOps = CollectPrecedingApplyOps(targetOp);

    if (precedingOps.empty()) {
        return failure();
    }

    mlir::SmallVector<stencil::ApplyOp, 12> involvedPrecedingOps;

    for (auto precedingOp : precedingOps) {
        auto mappedResults = MapPrecedingOpResults(precedingOp, targetOp);
        const auto last = std::remove_if(mappedResults.begin(), mappedResults.end(), [&targetOp](const auto& mapping) {
            return !IsApplyOpInputFusable(targetOp, mapping.second);
        });
        mappedResults.erase(last, mappedResults.end());

        const auto precedingStencil = mlir::SymbolTable::lookupNearestSymbolFrom<stencil::StencilOp>(precedingOp, precedingOp.getCalleeAttr());
        const auto targetStencil = mlir::SymbolTable::lookupNearestSymbolFrom<stencil::StencilOp>(targetOp, targetOp.getCalleeAttr());
        const float cost = FusePrecedingStencilCost(precedingStencil, targetStencil, mappedResults);
        if (cost > 0) {
            continue;
        }

        auto maybeFusedApplyOp = FusePrecedingApplyOp(precedingOp, targetOp, mappedResults, rewriter);
        if (targetOp != primaryTargetOp) {
            rewriter.eraseOp(targetOp);
        }
        if (failed(maybeFusedApplyOp)) {
            return mlir::failure();
        }
        targetOp = maybeFusedApplyOp.value();
        involvedPrecedingOps.push_back(precedingOp);
    }

    return std::pair{ targetOp, std::move(involvedPrecedingOps) };
}


class FuseElementwiseOps : public OpRewritePattern<stencil::ApplyOp> {
public:
    FuseElementwiseOps(MLIRContext* context,
                       PatternBenefit benefit = 1)
        : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}

    LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                  PatternRewriter& rewriter) const override {
        auto maybeFusedOp = FusePrecedingApplies(applyOp, rewriter);
        if (succeeded(maybeFusedOp)) {
            auto [fusedOp, precedingOps] = maybeFusedOp.value();
            rewriter.replaceOp(applyOp, fusedOp->getResults());
            for (auto precedingOp : precedingOps) {
                auto results = precedingOp->getResults();
                if (std::all_of(results.begin(), results.end(), [](mlir::Value result) { return result.getUses().empty(); })) {
                    rewriter.eraseOp(precedingOp);
                }
            }
            return success();
        }
        return failure();
    }
};


void FuseApplyOpsPass::runOnOperation() {
    Operation* op = getOperation();
    MLIRContext* context = op->getContext();

    RewritePatternSet patterns(context);
    patterns.add<FuseElementwiseOps>(context);

    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns), grc);

    mlir::PassManager pm{ &getContext() };
    pm.addPass(createSymbolDCEPass());
    if (pm.run(op).failed()) {
        signalPassFailure();
    }
}