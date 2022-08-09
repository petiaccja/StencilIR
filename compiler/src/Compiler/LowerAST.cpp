#include "LowerAST.hpp"

#include "AST/AST.hpp"
#include "AST/Types.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <ranges>
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


using ASTToMLIRTranformer = Transformer<ast::Node, mlir::Operation*>;

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
            const auto shape = llvm::ArrayRef<int64_t>{ 0 };
            return mlir::MemRefType::get(shape, elementType);
        }
    } visitor{ builder };
    return std::visit(visitor, type);
}

struct ASTToMLIRRules {
    mlir::OpBuilder& builder;

    mlir::ModuleOp operator()(const ASTToMLIRTranformer& tf, const ast::Module& module) const {
        auto moduleOp = builder.create<mlir::ModuleOp>(ConvertLocation(builder, module.location));

        builder.setInsertionPointToEnd(moduleOp.getBody());

        for (auto& kernel : module.kernels) {
            tf(*kernel);
        }

        auto mainFuncType = builder.getFunctionType(llvm::ArrayRef<mlir::Type>{}, llvm::ArrayRef<mlir::Type>{});
        auto mainFuncOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", mainFuncType);
        auto& mainBlock = mainFuncOp.getBody().emplaceBlock();

        builder.setInsertionPointToStart(&mainBlock);
        for (auto& statement : module.body) {
            tf(*statement);
        }

        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

        return moduleOp;
    }

    template <class T>
    auto operator()(const ASTToMLIRTranformer& tf, const ast::Constant<T>& constant) const {
        if constexpr (std::is_floating_point_v<T>) {
            const auto type = std::is_same_v<T, float> ? builder.getF32Type() : builder.getF64Type();
            return builder.create<mlir::arith::ConstantFloatOp>(ConvertLocation(builder, constant.location),
                                                                mlir::APFloat(constant.value),
                                                                type);
        }
        else if constexpr (std::is_integral_v<T>) {
            constexpr int numBits = sizeof(T) * 8;
            const auto type = constant.type ? ConvertType(builder, constant.type.value()) : mlir::Type(builder.getIntegerType(numBits));
            using Unsigned = std::make_unsigned_t<decltype(constant.value)>;
            const uint64_t unsignedValue = std::bit_cast<Unsigned>(constant.value);
            return builder.create<mlir::arith::ConstantOp>(ConvertLocation(builder, constant.location),
                                                           builder.getIntegerAttr(type, std::bit_cast<int64_t>(unsignedValue)),
                                                           type);
        }
        throw std::invalid_argument("Cannot lower this constant type.");
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Print& print) const {
        mlir::Value arg = tf(*print.argument)->getResult(0);
        return builder.create<mock::PrintOp>(ConvertLocation(builder, print.location),
                                             arg);
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Add& add) const {
        mlir::Value lhs = tf(*add.lhs)->getResult(0);
        mlir::Value rhs = tf(*add.rhs)->getResult(0);
        return builder.create<mlir::arith::AddFOp>(ConvertLocation(builder, add.location),
                                                   lhs,
                                                   rhs);
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::KernelFunc& kernelFunc) const {
        const auto symName = builder.getStringAttr(kernelFunc.name);
        mlir::ArrayRef<mlir::Type> inputTypes{};
        for (auto& param : kernelFunc.parameters) {
            inputTypes.vec().push_back(ConvertType(builder, param.second));
        }
        mlir::ArrayRef<mlir::Type> resultTypes{};
        for (auto& result : kernelFunc.results) {
            inputTypes.vec().push_back(ConvertType(builder, result));
        }
        const auto functionType = builder.getFunctionType(mlir::TypeRange{ inputTypes }, mlir::TypeRange{ resultTypes });
        const auto loc = ConvertLocation(builder, kernelFunc.location);
        auto kernelFuncOp = builder.create<mock::KernelFuncOp>(loc,
                                                               symName,
                                                               functionType);
        const auto previousBlock = builder.getBlock();
        const auto previousInsertionPoint = builder.getInsertionPoint();
        auto& kernelFuncBody = kernelFuncOp.getBody();
        auto& kernelFuncBlock = kernelFuncBody.emplaceBlock();
        builder.setInsertionPointToStart(&kernelFuncBlock);

        for (auto& statement : kernelFunc.body) {
            tf(*statement);
        }
        builder.create<mock::KernelReturnOp>(loc);

        builder.setInsertionPoint(previousBlock, previousInsertionPoint);

        return kernelFuncOp;
    }

    auto operator()(const ASTToMLIRTranformer& tf, const ast::KernelLaunch& kernelLaunch) const {
        const auto callee = mlir::StringRef(kernelLaunch.callee);

        std::vector<mlir::Value> gridDim;
        for (auto& gridAxis : kernelLaunch.gridDim) {
            const auto op = tf(*gridAxis);
            gridDim.push_back(op->getResult(0));
        }

        std::vector<mlir::Value> targets = {};

        std::vector<mlir::Value> operands;
        for (auto& operand : kernelLaunch.arguments) {
            const auto op = tf(*operand);
            operands.push_back(op->getResult(0));
        }

        return builder.create<mock::KernelCallOp>(ConvertLocation(builder, kernelLaunch.location),
                                                  callee,
                                                  mlir::ValueRange{ llvm::ArrayRef<mlir::Value>{ gridDim } },
                                                  mlir::ValueRange{ llvm::ArrayRef<mlir::Value>{ targets } },
                                                  mlir::ValueRange{ llvm::ArrayRef<mlir::Value>{ operands } });
    }
};

mlir::ModuleOp LowerAST(mlir::MLIRContext& context, ast::Module& node) {
    mlir::OpBuilder builder{ &context };
    ASTToMLIRTranformer transformer;
    ASTToMLIRRules rules{ builder };

    transformer.AddNodeTransformer<ast::Module>(rules);
    transformer.AddNodeTransformer<ast::Print>(rules);
    transformer.AddNodeTransformer<ast::Add>(rules);
    transformer.AddNodeTransformer<ast::KernelFunc>(rules);
    transformer.AddNodeTransformer<ast::KernelLaunch>(rules);

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
    context.getOrLoadDialect<mock::MockDialect>();

    auto moduleOp = transformer(node);
    return mlir::cast<mlir::ModuleOp>(moduleOp);
}