#include "FuseApplyOps.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <regex>


using namespace mlir;


bool CanFuseOperand(stencil::ApplyOp applyOp, size_t operandIdx) {
    const mlir::Value input = applyOp.getInputs()[operandIdx];
    const auto definingOp = input.getDefiningOp();
    const bool isDefinedByApply = definingOp && mlir::isa<stencil::ApplyOp>(definingOp);

    const stencil::StencilOp stencil = mlir::SymbolTable::lookupNearestSymbolFrom<stencil::StencilOp>(applyOp, applyOp.getCalleeAttr());
    const mlir::Value blockArg = stencil->getRegion(0).getBlocks().begin()->getArgument(operandIdx);
    const auto blockArgUsers = blockArg.getUsers();
    const bool isOnlyUsedBySample = std::all_of(blockArgUsers.begin(), blockArgUsers.end(), [](auto&& user) {
        return mlir::isa<stencil::SampleOp>(user);
    });

    return isDefinedByApply && isOnlyUsedBySample;
}


float FusionCost([[maybe_unused]] stencil::ApplyOp applyOp, [[maybe_unused]] size_t operandIdx) {
    return -1.0f;
}


mlir::StringAttr GetFusedSymName(stencil::StencilOp originalStencil, mlir::PatternRewriter& rewriter) {
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


void FuseOperand(stencil::ApplyOp applyOp, size_t operandIdx, mlir::PatternRewriter& rewriter) {
    // Get defining apply op
    mlir::Value input = applyOp.getInputs()[operandIdx];
    auto definingOp = mlir::dyn_cast<stencil::ApplyOp>(input.getDefiningOp());
    size_t definingIdx = -1;
    for (size_t i = 0; i < definingOp->getNumResults(); ++i) {
        if (input == definingOp->getResult(i)) {
            definingIdx = i;
            break;
        }
    }
    stencil::StencilOp definingStencil = mlir::SymbolTable::lookupNearestSymbolFrom<stencil::StencilOp>(definingOp, definingOp.getCalleeAttr());

    // Create fused stencil
    stencil::StencilOp originalStencil = mlir::SymbolTable::lookupNearestSymbolFrom<stencil::StencilOp>(applyOp, applyOp.getCalleeAttr());
    rewriter.setInsertionPointAfter(originalStencil);
    stencil::StencilOp fusedStencil = mlir::dyn_cast<stencil::StencilOp>(rewriter.clone(*originalStencil));
    fusedStencil.setSymNameAttr(GetFusedSymName(originalStencil, rewriter));
    auto originalFunctionType = originalStencil.getFunctionType();

    // Insert block arguments of defining stencil into fused stencil
    auto& fusedBlock = fusedStencil.getRegion().getBlocks().front();
    mlir::SmallVector<mlir::Type, 16> fusedInputTypes{ originalFunctionType.getInputs().begin(),
                                                       originalFunctionType.getInputs().end() };
    mlir::SmallVector<mlir::BlockArgument, 6> newBlockArgs;
    for (const auto& definingArg : definingStencil.getRegion().getBlocks().front().getArguments()) {
        const auto insertPos = definingArg.getArgNumber() + operandIdx + 1;
        auto newBlockArg = fusedBlock.insertArgument(
            insertPos,
            definingArg.getType(),
            definingArg.getLoc());
        fusedInputTypes.insert(fusedInputTypes.begin() + insertPos, definingArg.getType());
        newBlockArgs.push_back(std::move(newBlockArg));
    }
    fusedInputTypes.erase(fusedInputTypes.begin() + operandIdx);

    auto fusedFunctionType = rewriter.getFunctionType(fusedInputTypes, originalFunctionType.getResults());
    fusedStencil.setFunctionTypeAttr(mlir::TypeAttr::get(fusedFunctionType));

    // Replace sample ops with invoke ops
    auto sampledArg = fusedBlock.getArgument(operandIdx);
    mlir::SmallVector<stencil::SampleOp, 12> sampleOps;
    for (const auto& user : sampledArg.getUsers()) {
        auto sampleOp = mlir::dyn_cast<stencil::SampleOp>(user);
        assert(sampleOp);
        sampleOps.push_back(std::move(sampleOp));
    }

    mlir::SmallVector<mlir::Value, 6> invokeOpArgs;
    for (auto& blockArg : newBlockArgs) {
        invokeOpArgs.push_back(blockArg);
    }
    for (auto& sampleOp : sampleOps) {
        rewriter.setInsertionPointAfter(sampleOp);

        const mlir::Value index = sampleOp.getIndex();
        const auto invokeOp = rewriter.create<stencil::InvokeOp>(sampleOp->getLoc(),
                                                                 definingStencil.getFunctionType().getResults(),
                                                                 definingStencil.getSymName(),
                                                                 index,
                                                                 invokeOpArgs);
        rewriter.replaceOp(sampleOp, invokeOp->getResults()[definingIdx]);
    }

    // Erase obsolete argument from fused block
    fusedBlock.eraseArgument(operandIdx);

    // Replace apply op
    mlir::SmallVector<mlir::Value, 12> newInputs;
    std::copy_n(applyOp.getInputs().begin(), operandIdx, std::back_inserter(newInputs));
    std::copy(definingOp.getInputs().begin(), definingOp.getInputs().end(), std::back_inserter(newInputs));
    std::copy(applyOp.getInputs().begin() + operandIdx + 1, applyOp.getInputs().end(), std::back_inserter(newInputs));

    rewriter.setInsertionPointAfter(applyOp);
    rewriter.replaceOpWithNewOp<stencil::ApplyOp>(applyOp,
                                                  applyOp->getResultTypes(),
                                                  fusedStencil.getSymName(),
                                                  newInputs,
                                                  applyOp.getOutputs(),
                                                  applyOp.getOffsets(),
                                                  applyOp.getStaticOffsetsAttr());

    // Erase defining apply op if its results have no more uses
    if (std::all_of(definingOp->getResults().begin(), definingOp->getResults().end(), [](mlir::Value result) {
            return result.getUses().empty();
        })) {
        rewriter.eraseOp(definingOp);
    }
}


class FuseElementwiseOps : public OpRewritePattern<stencil::ApplyOp> {
public:
    FuseElementwiseOps(MLIRContext* context,
                       PatternBenefit benefit = 1)
        : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}

    LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                  PatternRewriter& rewriter) const override {
        std::vector<size_t> fusableOperands;
        for (size_t operandIdx = 0; operandIdx < applyOp.getInputs().size(); ++operandIdx) {
            if (CanFuseOperand(applyOp, operandIdx)
                && FusionCost(applyOp, operandIdx) < 0) {
                fusableOperands.push_back(operandIdx);
            }
        }

        if (fusableOperands.empty()) {
            return failure();
        }

        FuseOperand(applyOp, fusableOperands.front(), rewriter);

        return success();
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