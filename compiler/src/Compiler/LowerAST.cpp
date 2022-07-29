#include "LowerAST.hpp"

#include "AST/AST.hpp"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include <AST/Node.hpp>
#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <concepts>
#include <functional>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
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

struct ASTToMLIRRules {
    mlir::OpBuilder& builder;

    mlir::ModuleOp operator()(const ASTToMLIRTranformer& tf, const ast::Module& module) const {
        auto moduleOp = builder.create<mlir::ModuleOp>(ConvertLocation(builder, module.location));

        builder.setInsertionPointToEnd(moduleOp.getBody());

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

    auto operator()(const ASTToMLIRTranformer& tf, const ast::Constant<float>& constant) const {
        return builder.create<mlir::arith::ConstantFloatOp>(ConvertLocation(builder, constant.location),
                                                            mlir::APFloat(constant.value),
                                                            builder.getF32Type());
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
};

mlir::ModuleOp LowerAST(mlir::MLIRContext& context, ast::Module& node) {
    mlir::OpBuilder builder{ &context };
    ASTToMLIRTranformer transformer;
    ASTToMLIRRules rules{ builder };

    transformer.AddNodeTransformer<ast::Module>(rules);
    transformer.AddNodeTransformer<ast::Constant<float>>(rules);
    transformer.AddNodeTransformer<ast::Print>(rules);
    transformer.AddNodeTransformer<ast::Add>(rules);

    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
    context.getOrLoadDialect<mock::MockDialect>();

    auto moduleOp = transformer(node);
    return mlir::cast<mlir::ModuleOp>(moduleOp);
}