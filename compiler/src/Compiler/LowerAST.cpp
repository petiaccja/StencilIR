#include "LowerAST.hpp"

#include "AST/AST.hpp"
#include "AST/Types.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include <AST/Node.hpp>
#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <list>
#include <llvm/ADT/ScopedHashTable.h>
#include <memory>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <ranges>
#include <stack>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

using namespace std::string_literals;


template <class SourceT, class TargetT>
class Transformer {
public:
    Transformer() = default;
    Transformer(Transformer&&) = delete;
    Transformer& operator=(Transformer&&) = delete;
    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;

    template <class NodeT, class TransformFun>
    requires std::invocable<TransformFun, const Transformer&, const NodeT&> && std::is_base_of_v<SourceT, NodeT>
    void AddNodeTransformer(TransformFun transformer) {
        auto wrapper = [this, transformer = std::move(transformer)](const SourceT& source) {
            return transformer(*this, dynamic_cast<const NodeT&>(source));
        };
        m_transformers.insert_or_assign(typeid(NodeT), wrapper);
    }

    TargetT operator()(const SourceT& source) const {
        const std::type_index dynamicType = typeid(source);
        const auto transformerIt = m_transformers.find(dynamicType);
        if (transformerIt == m_transformers.end()) {
            throw std::invalid_argument("No transformer added for type "s + dynamicType.name());
        }
        return transformerIt->second(source);
    }

private:
    std::unordered_map<std::type_index, std::function<TargetT(const SourceT&)>> m_transformers;
};


template <class Key, class Value>
class SymbolTable {
public:
    std::unordered_map<Key, Value>& Push() {
        m_scopes.push_back({});
        return m_scopes.back();
    }
    void Pop() {
        m_scopes.pop_back();
    }
    void Insert(const Key& key, Value value) {
        assert(!m_scopes.empty());
        if (m_scopes.back().contains(key)) {
            throw std::invalid_argument("already defined");
        }
        m_scopes.back().insert_or_assign(key, value);
    }
    const Value& Lookup(const Key& key) const {
        return TryLookup(key).value();
    }
    bool Contains(const Key& key) const {
        return TryLookup(key).has_value();
    }

private:
    std::optional<std::reference_wrapper<const Value>> TryLookup(const Key& key) const {
        for (auto scopeIt = m_scopes.rbegin(); scopeIt != m_scopes.rend(); ++scopeIt) {
            auto recordIt = scopeIt->find(key);
            if (recordIt != scopeIt->end()) {
                return { std::ref(recordIt->second) };
            }
        }
        return {};
    }

private:
    std::list<std::unordered_map<Key, Value>> m_scopes;
};

using ASTToMLIRTranformer = Transformer<ast::Node, std::vector<mlir::Value>>;

mlir::Location ConvertLocation(mlir::OpBuilder& builder, const std::optional<ast::Location>& location) {
    if (location) {
        auto fileattr = builder.getStringAttr(location->file);
        return mlir::FileLineColLoc::get(fileattr, location->line, location->col);
    }
    return builder.getUnknownLoc();
}

mlir::Type ConvertType(mlir::OpBuilder& builder, types::Type type) {
    struct {
        mlir::OpBuilder& builder;
        mlir::Type operator()(const types::FundamentalType& type) const {
            switch (type.type) {
                case types::FundamentalType::SINT8: return builder.getIntegerType(8, true);
                case types::FundamentalType::SINT16: return builder.getIntegerType(16, true);
                case types::FundamentalType::SINT32: return builder.getIntegerType(32, true);
                case types::FundamentalType::SINT64: return builder.getIntegerType(64, true);
                case types::FundamentalType::UINT8: return builder.getIntegerType(8, false);
                case types::FundamentalType::UINT16: return builder.getIntegerType(16, false);
                case types::FundamentalType::UINT32: return builder.getIntegerType(32, false);
                case types::FundamentalType::UINT64: return builder.getIntegerType(64, false);
                case types::FundamentalType::SSIZE: return builder.getIndexType();
                case types::FundamentalType::USIZE: return builder.getIndexType();
                case types::FundamentalType::FLOAT32: return builder.getF32Type();
                case types::FundamentalType::FLOAT64: return builder.getF64Type();
                case types::FundamentalType::BOOL: return builder.getI1Type();
            }
            throw std::invalid_argument("Unknown type.");
        }
        mlir::Type operator()(const types::FieldType& type) const {
            const mlir::Type elementType = (*this)(type.elementType);

            constexpr auto offset = mlir::ShapedType::kDynamicStrideOrOffset;
            std::vector<int64_t> shape(type.numDimensions, mlir::ShapedType::kDynamicSize);
            std::vector<int64_t> strides(type.numDimensions, mlir::ShapedType::kDynamicStrideOrOffset);
            auto strideMap = mlir::makeStridedLinearLayoutMap(strides, offset, builder.getContext());

            return mlir::MemRefType::get(shape, elementType, strideMap);
        }
    } visitor{ builder };
    return std::visit(visitor, type);
}

struct ASTToMLIRRules {
    mlir::OpBuilder& builder;
    SymbolTable<std::string, mlir::Value>& symbolTable;
    std::optional<mlir::ModuleOp>& moduleOp;
    std::shared_ptr<const ast::KernelFunc>& currentFunc;

    template <class Func>
    void InsertInBlock(mlir::Block& block, Func func) const {
        auto previousInsertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(&block);
        func();
        builder.restoreInsertionPoint(previousInsertionPoint);
    }

    //--------------------------------------------------------------------------
    // Arithmetic and misc
    //--------------------------------------------------------------------------

    template <class T>
    auto operator()(const ASTToMLIRTranformer& tf, const ast::Constant<T>& node) const -> std::vector<mlir::Value> {
        if constexpr (std::is_floating_point_v<T>) {
            const auto type = std::is_same_v<T, float> ? builder.getF32Type() : builder.getF64Type();
            auto op = builder.create<mlir::arith::ConstantFloatOp>(ConvertLocation(builder, node.location),
                                                                   mlir::APFloat(node.value),
                                                                   type);
            return { op->getResult(0) };
        }
        else if constexpr (std::is_integral_v<T>) {
            constexpr int numBits = sizeof(T) * 8;
            const auto type = node.type ? ConvertType(builder, node.type.value()) : mlir::Type(builder.getIntegerType(numBits));
            using Unsigned = std::make_unsigned_t<decltype(node.value)>;
            const uint64_t unsignedValue = std::bit_cast<Unsigned>(node.value);
            auto op = builder.create<mlir::arith::ConstantOp>(ConvertLocation(builder, node.location),
                                                              builder.getIntegerAttr(type, std::bit_cast<int64_t>(unsignedValue)),
                                                              type);
            return { op->getResult(0) };
        }
        throw std::invalid_argument("Cannot lower this constant type.");
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Print& node) const -> std::vector<mlir::Value> {
        mlir::Value arg = tf(*node.argument).front();
        builder.create<mock::PrintOp>(ConvertLocation(builder, node.location),
                                      arg);
        return {};
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Add& node) const -> std::vector<mlir::Value> {
        mlir::Value lhs = tf(*node.lhs).front();
        mlir::Value rhs = tf(*node.rhs).front();
        auto op = builder.create<mlir::arith::AddFOp>(ConvertLocation(builder, node.location),
                                                      lhs,
                                                      rhs);
        return { op->getResult(0) };
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Mul& node) const -> std::vector<mlir::Value> {
        mlir::Value lhs = tf(*node.lhs).front();
        mlir::Value rhs = tf(*node.rhs).front();
        auto op = builder.create<mlir::arith::MulFOp>(ConvertLocation(builder, node.location),
                                                      lhs,
                                                      rhs);
        return { op->getResult(0) };
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::ReshapeField& node) const -> std::vector<mlir::Value> {
        auto loc = ConvertLocation(builder, node.location);

        mlir::Value source = tf(*node.field).front();
        mlir::MemRefType sourceType = source.getType().dyn_cast<mlir::MemRefType>();

        constexpr auto offsetType = mlir::ShapedType::kDynamicStrideOrOffset;
        std::vector<int64_t> shapeType(node.shape.size(), mlir::ShapedType::kDynamicSize);
        std::vector<int64_t> stridesType(node.shape.size(), mlir::ShapedType::kDynamicStrideOrOffset);
        auto strideMap = mlir::makeStridedLinearLayoutMap(stridesType, offsetType, builder.getContext());

        mlir::MemRefType type = mlir::MemRefType::get(shapeType, sourceType.getElementType(), strideMap);
        mlir::Value offset = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        std::vector<mlir::Value> shape;
        std::vector<mlir::Value> strides;

        for (auto& shapeExpr : node.shape) {
            shape.push_back(tf(*shapeExpr).front());
        }

        for (auto& strideExpr : node.strides) {
            strides.push_back(tf(*strideExpr).front());
        }

        mlir::Value result = builder.create<mlir::memref::ReinterpretCastOp>(loc, type, source, offset, shape, strides);
        return { result };
    }

    //--------------------------------------------------------------------------
    // Structure
    //--------------------------------------------------------------------------

    auto operator()(const ASTToMLIRTranformer& tf, const ast::KernelFunc& node) const -> std::vector<mlir::Value> {
        currentFunc = std::dynamic_pointer_cast<const ast::KernelFunc>(node.shared_from_this());

        // Create operation
        std::vector<mlir::Type> inputTypes{};
        for (auto& param : node.parameters) {
            inputTypes.push_back(ConvertType(builder, param.type));
        }
        std::vector<mlir::Type> resultTypes{};
        for (auto& result : node.results) {
            resultTypes.push_back(ConvertType(builder, result));
        }
        const auto functionType = builder.getFunctionType(mlir::TypeRange{ inputTypes }, mlir::TypeRange{ resultTypes });
        const auto loc = ConvertLocation(builder, node.location);
        auto kernelFuncOp = builder.create<mock::KernelFuncOp>(loc,
                                                               node.name,
                                                               functionType,
                                                               mlir::APInt(64, node.numDimensions));

        // Create function body
        auto& kernelFuncBlock = *kernelFuncOp.addEntryBlock();

        symbolTable.Push();
        for (size_t i = 0; i < node.parameters.size(); ++i) {
            symbolTable.Insert(node.parameters[i].name, kernelFuncBlock.getArgument(i));
        }

        InsertInBlock(kernelFuncBlock, [&] {
            for (auto& statement : node.body) {
                tf(*statement);
            }
        });

        symbolTable.Pop();

        return {};
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::KernelReturn& node) const -> std::vector<mlir::Value> {
        const auto loc = ConvertLocation(builder, node.location);
        std::vector<mlir::Value> values;
        for (const auto& value : node.values) {
            for (auto& item : tf(*value)) {
                values.push_back(item);
            }
        }
        builder.create<mock::KernelReturnOp>(loc, values);
        return {};
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::KernelLaunch& node) const -> std::vector<mlir::Value> {
        const auto callee = mlir::StringRef(node.callee);

        std::vector<mlir::Value> gridDim;
        for (auto& gridAxis : node.gridDim) {
            gridDim.push_back(tf(*gridAxis).front());
        }

        std::vector<mlir::Value> targets;
        for (auto& target : node.targets) {
            targets.push_back(tf(*target).front());
        }

        std::vector<mlir::Value> operands;
        for (auto& argument : node.arguments) {
            operands.push_back(tf(*argument).front());
        }

        builder.create<mock::KernelLaunchOp>(ConvertLocation(builder, node.location),
                                             callee,
                                             mlir::ValueRange{ llvm::ArrayRef<mlir::Value>{ gridDim } },
                                             mlir::ValueRange{ llvm::ArrayRef<mlir::Value>{ targets } },
                                             mlir::ValueRange{ llvm::ArrayRef<mlir::Value>{ operands } });

        return {};
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Module& node) const -> std::vector<mlir::Value> {
        auto moduleOp = builder.create<mlir::ModuleOp>(ConvertLocation(builder, node.location));
        this->moduleOp = moduleOp;

        builder.setInsertionPointToEnd(moduleOp.getBody());

        // Render kernels
        for (auto& kernel : node.kernels) {
            tf(*kernel);
        }

        // Create a main function.
        std::vector<mlir::Type> inputTypes{};
        for (auto& param : node.parameters) {
            inputTypes.push_back(ConvertType(builder, param.type));
        }
        const auto mainFuncType = builder.getFunctionType(mlir::TypeRange{ inputTypes }, mlir::TypeRange{});
        auto mainFuncOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", mainFuncType);

        // Create body.
        mlir::Block& mainBlock = *mainFuncOp.addEntryBlock();
        symbolTable.Push();
        for (size_t i = 0; i < node.parameters.size(); ++i) {
            symbolTable.Insert(node.parameters[i].name, mainBlock.getArgument(i));
        }

        InsertInBlock(mainBlock, [&] {
            for (auto& statement : node.body) {
                tf(*statement);
            }
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        });

        symbolTable.Pop();

        return {};
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::SymbolRef& node) const -> std::vector<mlir::Value> {
        return { symbolTable.Lookup(node.name) };
    }

    //--------------------------------------------------------------------------
    // Kernel intrinsics
    //--------------------------------------------------------------------------

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Index& node) const -> std::vector<mlir::Value> {
        const auto loc = ConvertLocation(builder, node.location);

        const int64_t numDims = currentFunc->numDimensions;
        const auto indexType = mlir::MemRefType::get({ numDims }, builder.getIndexType());

        auto op = builder.create<mock::IndexOp>(loc, indexType);
        return { op.getIndex() };
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Offset& node) const -> std::vector<mlir::Value> {
        const auto loc = ConvertLocation(builder, node.location);
        const auto index = tf(*node.index).front();
        const auto offset = builder.getI64ArrayAttr(node.offset);
        auto op = builder.create<mock::OffsetOp>(loc, index.getType(), index, offset);
        return { op.getOffsetedIndex() };
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Sample& node) const -> std::vector<mlir::Value> {
        const auto loc = ConvertLocation(builder, node.location);
        const auto field = tf(*node.field).front();
        const auto index = tf(*node.index).front();

        auto fieldType = field.getType();
        if (!fieldType.isa<mlir::MemRefType>()) {
            throw std::invalid_argument("SampleOp must be used to sample fields.");
        }
        auto elementType = fieldType.dyn_cast<mlir::MemRefType>().getElementType();

        auto op = builder.create<mock::SampleOp>(loc, elementType, field, index);
        return { op.getSampledValue() };
    }
};

mlir::ModuleOp LowerAST(mlir::MLIRContext& context, const ast::Module& node) {
    mlir::OpBuilder builder{ &context };
    SymbolTable<std::string, mlir::Value> symbolTable;
    std::optional<mlir::ModuleOp> moduleOp;
    std::shared_ptr<const ast::KernelFunc> currentFunc;
    ASTToMLIRTranformer transformer;
    ASTToMLIRRules rules{ builder, symbolTable, moduleOp, currentFunc };

    transformer.AddNodeTransformer<ast::Print>(rules);
    transformer.AddNodeTransformer<ast::Add>(rules);
    transformer.AddNodeTransformer<ast::Mul>(rules);
    transformer.AddNodeTransformer<ast::ReshapeField>(rules);

    transformer.AddNodeTransformer<ast::Module>(rules);
    transformer.AddNodeTransformer<ast::KernelFunc>(rules);
    transformer.AddNodeTransformer<ast::KernelReturn>(rules);
    transformer.AddNodeTransformer<ast::KernelLaunch>(rules);
    transformer.AddNodeTransformer<ast::SymbolRef>(rules);

    transformer.AddNodeTransformer<ast::Index>(rules);
    transformer.AddNodeTransformer<ast::Offset>(rules);
    transformer.AddNodeTransformer<ast::Sample>(rules);

    transformer.AddNodeTransformer<ast::Constant<float>>(rules);
    transformer.AddNodeTransformer<ast::Constant<double>>(rules);
    transformer.AddNodeTransformer<ast::Constant<int8_t>>(rules);
    transformer.AddNodeTransformer<ast::Constant<int16_t>>(rules);
    transformer.AddNodeTransformer<ast::Constant<int32_t>>(rules);
    transformer.AddNodeTransformer<ast::Constant<int64_t>>(rules);
    transformer.AddNodeTransformer<ast::Constant<uint8_t>>(rules);
    transformer.AddNodeTransformer<ast::Constant<uint16_t>>(rules);
    transformer.AddNodeTransformer<ast::Constant<uint32_t>>(rules);
    transformer.AddNodeTransformer<ast::Constant<uint64_t>>(rules);

    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mock::MockDialect>();

    transformer(node);
    return moduleOp.value();
}