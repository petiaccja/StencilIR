#include "StencilOps.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/InliningUtils.h>

// clang-format: off
#include <Stencil/IR/StencilDialect.cpp.inc>
#define GET_OP_CLASSES
#include <Stencil/IR/Stencil.cpp.inc>
// clang-format: on


namespace stencil {
using namespace mlir;


class StencilInlinerInterface;


void StencilDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <Stencil/IR/Stencil.cpp.inc>
        >();
    addInterfaces<StencilInlinerInterface>();
}


//------------------------------------------------------------------------------
// Inliner interface
//------------------------------------------------------------------------------

class StencilInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation* call, Operation* callable, bool wouldBeCloned) const override {
        return true;
    }

    bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
        return true;
    }

    bool isLegalToInline(Operation* op, Region* dest, bool wouldBeCloned, IRMapping& valueMapping) const override {
        return true;
    }

    void handleTerminator(Operation* op, Block* newDest) const override {
        llvm_unreachable("must implement handleTerminator in the case of multiple inlined blocks");
    }

    void handleTerminator(Operation* op, ArrayRef<Value> valuesToReplace) const override {
        auto returnOp = cast<ReturnOp>(op);

        assert(returnOp.getNumOperands() == valuesToReplace.size());
        for (const auto& it : llvm::enumerate(returnOp.getOperands())) {
            valuesToReplace[it.index()].replaceAllUsesWith(it.value());
        }
    }

    Operation* materializeCallConversion(OpBuilder& builder,
                                         Value input,
                                         Type resultType,
                                         Location conversionLoc) const override {
        return nullptr;
    }

    void processInlinedCallBlocks(Operation* call, iterator_range<Region::iterator> inlinedBlocks) const override {
        auto invokeOp = cast<InvokeOp>(call);
        auto indexValue = invokeOp.getIndex();

        for (auto& block : inlinedBlocks) {
            block.walk([&](Operation* operation) {
                if (auto indexOp = dyn_cast<IndexOp>(operation)) {
                    indexOp.getResult().replaceAllUsesWith(indexValue);
                    indexOp->erase();
                }
                return WalkResult::advance();
            });
        }
    }
};


//------------------------------------------------------------------------------
// StencilOp
//------------------------------------------------------------------------------

StencilOp StencilOp::create(Location location,
                            StringRef name,
                            FunctionType type,
                            IntegerAttr numDimensions,
                            ArrayRef<NamedAttribute> attrs) {
    OpBuilder builder(location->getContext());
    OperationState state(location, getOperationName());
    StencilOp::build(builder, state, name, type, numDimensions, attrs);
    return cast<StencilOp>(Operation::create(state));
}


StencilOp StencilOp::create(Location location,
                            StringRef name,
                            FunctionType type,
                            IntegerAttr numDimensions,
                            Operation::dialect_attr_range attrs) {
    SmallVector<NamedAttribute, 8> attrRef(attrs);
    return create(location, name, type, numDimensions, llvm::ArrayRef(attrRef));
}


StencilOp StencilOp::create(Location location,
                            StringRef name,
                            FunctionType type,
                            IntegerAttr numDimensions,
                            ArrayRef<NamedAttribute> attrs,
                            ArrayRef<DictionaryAttr> argAttrs) {
    StencilOp func = create(location, name, type, numDimensions, attrs);
    func.setAllArgAttrs(argAttrs);
    return func;
}


void StencilOp::build(OpBuilder& builder,
                      OperationState& state,
                      StringRef name,
                      FunctionType type,
                      IntegerAttr numDimensions,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(name));
    state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
    state.addAttribute(getNumDimensionsAttrName(state.name), numDimensions);
    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();

    if (argAttrs.empty())
        return;
    assert(type.getNumInputs() == argAttrs.size());
    function_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
        getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}


ParseResult StencilOp::parse(OpAsmParser& parser, OperationState& result) {
    auto buildFuncType =
        [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
           function_interface_impl::VariadicFlag,
           std::string&) { return builder.getFunctionType(argTypes, results); };

    return function_interface_impl::parseFunctionOp(parser,
                                                    result,
                                                    /*allowVariadic=*/false,
                                                    getFunctionTypeAttrName(result.name),
                                                    buildFuncType,
                                                    getArgAttrsAttrName(result.name),
                                                    getResAttrsAttrName(result.name));
}

void StencilOp::print(OpAsmPrinter& p) {
    function_interface_impl::printFunctionOp(p,
                                             *this,
                                             /*isVariadic=*/false,
                                             getFunctionTypeAttrName(),
                                             getArgAttrsAttrName(),
                                             getResAttrsAttrName());
}


//------------------------------------------------------------------------------
// ApplyOp
//------------------------------------------------------------------------------

LogicalResult ApplyOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr) {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }
    StencilOp fn = symbolTable.lookupNearestSymbolFrom<StencilOp>(*this, fnAttr);
    if (!fn) {
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";
    }

    // Verify that the dimensions match up
    if (getNumResults() > 0) {
        const auto numStencilDims = fn.getNumDimensions();
        const auto numMyDims = getResultTypes()[0].dyn_cast<mlir::ShapedType>().getRank();
        if (numStencilDims != numMyDims) {
            return emitOpError() << "number of stencil dimensions (" << numStencilDims << ")"
                                 << " does not equal number of output dimensions (" << numMyDims << ")";
        }
    }

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getFunctionType();
    const size_t numCalleeParams = fnType.getNumInputs();
    const size_t numCalleeResults = fnType.getNumResults();
    const size_t numArgs = getInputs().size();
    const size_t numTargets = getOutputs().size();
    if (numCalleeParams != numArgs) {
        return emitOpError("number of arguments must match number of operands of callee");
    }
    if (numCalleeResults != numTargets) {
        return emitOpError("number of targets must match number of callee returns");
    }

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
        mlir::Type inputType = getInputs()[i].getType();
        mlir::Type calleeInputType = fnType.getInput(i);
        if (inputType != calleeInputType) {
            return emitOpError("operand type mismatch: expected operand type ")
                   << inputType << ", but callee's operand type is "
                   << calleeInputType << " for operand number " << i;
        }
    }

    if (fnType.getNumResults() != getOutputs().size()) {
        return emitOpError("incorrect number of results for callee");
    }

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
        mlir::Type calleeResultType = fnType.getResult(i);
        mlir::ShapedType targetType = getOutputs()[i].getType().dyn_cast<mlir::ShapedType>();
        if (targetType.getElementType() != calleeResultType) {
            auto diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getResultTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }
    }

    return success();
}


static mlir::SmallVector<mlir::Type, 8> InferApplyOpResultTypes(mlir::ValueRange outputs) {
    if (!outputs.empty()) {
        auto types = outputs.getTypes();
        if (types.front().isa<mlir::TensorType>()) {
            return { types.begin(), types.end() };
        }
    }
    return {};
}


void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs) {
    return build(odsBuilder,
                 odsState,
                 InferApplyOpResultTypes(outputs),
                 callee,
                 inputs,
                 outputs,
                 ::mlir::ValueRange{},
                 odsBuilder.getI64ArrayAttr({}));
}


void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs,
                    ::llvm::ArrayRef<int64_t> static_offsets) {
    return build(odsBuilder,
                 odsState,
                 InferApplyOpResultTypes(outputs),
                 callee,
                 inputs,
                 outputs,
                 ::mlir::ValueRange{},
                 odsBuilder.getI64ArrayAttr(static_offsets));
}


void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs,
                    ::mlir::ValueRange offsets) {
    return build(odsBuilder,
                 odsState,
                 InferApplyOpResultTypes(outputs),
                 callee,
                 inputs,
                 outputs,
                 offsets,
                 odsBuilder.getI64ArrayAttr({}));
}

void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs,
                    ::mlir::ValueRange offsets,
                    ::llvm::ArrayRef<int64_t> static_offsets) {
    return build(odsBuilder,
                 odsState,
                 InferApplyOpResultTypes(outputs),
                 callee,
                 inputs,
                 outputs,
                 offsets,
                 odsBuilder.getI64ArrayAttr(static_offsets));
}

::mlir::LogicalResult ApplyOp::verify() {
    const auto& outputTypes = getOutputs().getTypes();
    const auto& resultTypes = getResultTypes();

    // Either all output operands are tensors or memrefs
    bool isAllTensor = std::all_of(outputTypes.begin(), outputTypes.end(), [](mlir::Type type) {
        return type.isa<mlir::TensorType>();
    });
    bool isAllMemref = std::all_of(outputTypes.begin(), outputTypes.end(), [](mlir::Type type) {
        return type.isa<mlir::MemRefType>();
    });
    if (!isAllTensor && !isAllMemref) {
        return emitOpError("output operands must be either all tensors or all memrefs, not mixed");
    }

    if (isAllTensor) {
        // Same number of output operands and results
        if (outputTypes.size() != resultTypes.size()) {
            return emitOpError("must have equal number of output operands as results");
        }

        // Same types of output operands and results
        for (size_t i = 0; i < outputTypes.size(); ++i) {
            if (outputTypes[i].getTypeID() != resultTypes[i].getTypeID()) {
                return emitOpError("output operand must have the same types as results");
            }
        }
    }
    if (isAllMemref) {
        if (resultTypes.size() != 0) {
            return emitOpError("ops with memref semantics cannot have results");
        }
    }

    // Either static or dynamic offsets
    if (!getOffsets().empty() && !getStaticOffsets().empty()) {
        return emitOpError("cannot have both static and dynamic offsets");
    }

    return success();
}

void ApplyOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects) {
    for (const auto& input : getInputs()) {
        if (input.getType().isa<ShapedType>()) {
            effects.emplace_back(MemoryEffects::Read::get(), input, SideEffects::DefaultResource::get());
        }
    }
    for (const auto& output : getOutputs()) {
        effects.emplace_back(MemoryEffects::Write::get(), output, SideEffects::DefaultResource::get());
    }
    auto stencilAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    auto callee = SymbolTable::lookupNearestSymbolFrom(*this, stencilAttr.getAttr());
    if (auto stencil = mlir::dyn_cast<StencilOp>(callee)) {
        auto& body = stencil.getRegion();
        bool hasReadEffects = false;
        bool hasWriteEffects = false;
        body.walk([&](Operation* nestedOp) {
                if (auto effectInterface = mlir::dyn_cast<MemoryEffectOpInterface>(nestedOp)) {
                    hasReadEffects = hasReadEffects || effectInterface.hasEffect<MemoryEffects::Read>();
                    hasWriteEffects = hasWriteEffects || effectInterface.hasEffect<MemoryEffects::Write>();
                    hasWriteEffects = hasWriteEffects || nestedOp->hasTrait<::mlir::OpTrait::HasRecursiveMemoryEffects>();
                }
                else {
                    hasWriteEffects = true;
                }
                return !hasWriteEffects ? WalkResult::advance() : WalkResult::interrupt();
            })
            .wasInterrupted();
        if (hasReadEffects) {
            effects.emplace_back(MemoryEffects::Read::get());
        }
        if (hasWriteEffects) {
            effects.emplace_back(MemoryEffects::Write::get());
        }
    }
}


//------------------------------------------------------------------------------
// InvokeOp
//------------------------------------------------------------------------------

LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr)
        return emitOpError("requires a 'callee' symbol reference attribute");
    StencilOp fn = symbolTable.lookupNearestSymbolFrom<StencilOp>(*this, fnAttr);
    if (!fn)
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getFunctionType();
    if (fnType.getNumInputs() != getArguments().size())
        return emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
        if (getArguments()[i].getType() != fnType.getInput(i))
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided "
                   << getArguments()[i].getType() << " for operand number " << i;

    if (fnType.getNumResults() != getNumResults())
        return emitOpError("incorrect number of results for callee");

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
        if (getResult(i).getType() != fnType.getResult(i)) {
            auto diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getResultTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }

    return success();
}

FunctionType InvokeOp::getCalleeType() {
    std::vector<Type> argumentTypes;
    for (const auto& arg : getArguments()) {
        argumentTypes.push_back(arg.getType());
    }
    return FunctionType::get(getContext(), argumentTypes, getResultTypes());
}


//------------------------------------------------------------------------------
// IndexOp
//------------------------------------------------------------------------------

mlir::LogicalResult IndexOp::verify() {
    auto stencil = (*this)->getParentOfType<StencilOp>();
    if (!stencil) {
        return emitOpError() << "index op must be enclosed in a stencil";
    }
    auto resultType = getResult().getType().dyn_cast<VectorType>();
    const auto indexDims = resultType.getShape()[0];
    const auto stencilDims = stencil.getNumDimensions();
    if (stencilDims != indexDims) {
        return emitOpError() << "index op has dimension " << indexDims
                             << " but enclosing stencil has dimension " << stencilDims;
    }
    return success();
}

//------------------------------------------------------------------------------
// Index manipulation
//------------------------------------------------------------------------------

void ProjectOp::build(::mlir::OpBuilder& odsBuilder,
                      ::mlir::OperationState& odsState,
                      ::mlir::Value source,
                      ::mlir::ArrayRef<int64_t> positions) {
    std::array<int64_t, 1> shape{ int64_t(positions.size()) };
    auto resultType = mlir::VectorType::get(shape, odsBuilder.getIndexType());
    auto elementsAttr = odsBuilder.getI64ArrayAttr(positions);
    return build(odsBuilder, odsState, resultType, source, elementsAttr);
}


void ExtendOp::build(::mlir::OpBuilder& odsBuilder,
                     ::mlir::OperationState& odsState,
                     ::mlir::Value source,
                     int64_t position,
                     ::mlir::Value value) {
    auto inputType = source.getType().dyn_cast<mlir::VectorType>();
    assert(inputType);
    std::array<int64_t, 1> shape{ inputType.getShape()[0] + 1 };
    auto resultType = mlir::VectorType::get(shape, odsBuilder.getIndexType());
    auto dimensionAttr = odsBuilder.getIndexAttr(position);
    return build(odsBuilder, odsState, resultType, source, dimensionAttr, value);
}


void ExchangeOp::build(::mlir::OpBuilder& odsBuilder,
                       ::mlir::OperationState& odsState,
                       ::mlir::Value source,
                       int64_t position,
                       ::mlir::Value value) {
    auto resultType = source.getType();
    auto dimensionAttr = odsBuilder.getIndexAttr(position);
    return build(odsBuilder, odsState, resultType, source, dimensionAttr, value);
}


void ExtractOp::build(::mlir::OpBuilder& odsBuilder,
                      ::mlir::OperationState& odsState,
                      ::mlir::Value source,
                      int64_t position) {
    auto dimensionAttr = odsBuilder.getIndexAttr(position);
    return build(odsBuilder, odsState, odsBuilder.getIndexType(), source, dimensionAttr);
}


//------------------------------------------------------------------------------
// SampleOp
//------------------------------------------------------------------------------

mlir::LogicalResult SampleOp::verify() {
    mlir::Value field = getField();
    mlir::Value index = getIndex();

    auto fieldType = field.getType().dyn_cast<mlir::ShapedType>();
    auto indexType = index.getType().dyn_cast<mlir::VectorType>();
    assert(fieldType);
    assert(indexType);

    if (!fieldType.hasRank()) {
        return emitOpError("ranked shaped type expected for field");
    }
    if (!indexType.hasStaticShape()) {
        return emitOpError("index with a static shape is expected");
    }
    if (indexType.getShape().size() != 1) {
        return emitOpError("index must be a one-dimensional vector");
    }
    if (indexType.getElementType() != mlir::IndexType::get(getContext())) {
        return emitOpError("index must be a vector with elements of type 'index'");
    }
    if (fieldType.getRank() != indexType.getShape()[0]) {
        return emitOpError() << "field's rank of " << fieldType.getRank()
                             << " does not match index's size of " << indexType.getShape()[0];
    }

    return success();
}


//------------------------------------------------------------------------------
// Folding
//------------------------------------------------------------------------------

OpFoldResult JumpOp::fold(FoldAdaptor) {
    auto input = getInputIndex();
    auto offset = getOffset();
    auto range = offset.getAsRange<mlir::IntegerAttr>();
    if (std::all_of(range.begin(), range.end(), [](mlir::IntegerAttr attr) { return attr.getInt() == 0; })) {
        return input;
    }
    return getResult();
}


struct SimplifyJumpChain : public mlir::OpRewritePattern<JumpOp> {
    explicit SimplifyJumpChain(mlir::MLIRContext* context)
        : OpRewritePattern<JumpOp>(context, 1) {}

    mlir::LogicalResult matchAndRewrite(JumpOp op, mlir::PatternRewriter& rewriter) const override {
        auto input = op.getInputIndex();
        auto definingOp = input.getDefiningOp();
        if (definingOp) {
            if (auto definingJumpOp = mlir::dyn_cast<JumpOp>(definingOp)) {
                mlir::SmallVector<mlir::Attribute, 4> offsetSum;
                auto myOffset = op.getOffset();
                auto definingOffset = definingJumpOp.getOffset();

                auto myRange = myOffset.getAsRange<mlir::IntegerAttr>();
                auto definingRange = definingOffset.getAsRange<mlir::IntegerAttr>();
                assert(myOffset.size() == definingOffset.size());
                auto [myIt, defIt] = std::tuple{ myRange.begin(), definingRange.begin() };
                for (; myIt != myRange.end() && defIt != definingRange.end(); ++myIt, ++defIt) {
                    auto type = (*myIt).getType();
                    auto value = (*myIt).getInt() + (*defIt).getInt();
                    auto sum = mlir::IntegerAttr::get(type, value);
                    offsetSum.push_back(mlir::cast<mlir::Attribute>(sum));
                }

                assert(offsetSum.size() == myOffset.size());

                rewriter.replaceOpWithNewOp<JumpOp>(op,
                                                    op->getResultTypes()[0],
                                                    definingJumpOp.getInputIndex(),
                                                    mlir::ArrayAttr::get(getContext(), offsetSum));
                return success();
            }
        }
        return mlir::failure();
    }
};

void JumpOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<SimplifyJumpChain>(context);
}


OpFoldResult ProjectOp::fold(FoldAdaptor) {
    const auto positions = getPositions();
    const auto range = positions.getAsRange<mlir::IntegerAttr>();
    int64_t idx = 0;
    for (const auto& pos : range) {
        if (pos.getInt() != idx++) {
            return getResult();
        }
    }
    return getSource();
}


} // namespace stencil