include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"


def Mock_Dialect : Dialect {
  let name = "mock";
  let cppNamespace = "::mock";
  let hasConstantMaterializer = 0;
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class Mock_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mock_Dialect, mnemonic, traits>;


def ConstantOp : Mock_Op<"constant", [NoSideEffect]> {
  let summary = "constant";
  let arguments = (ins F32Attr:$value);
  let results = (outs F32:$output);
}


def PrintOp : Mock_Op<"print", [NoSideEffect]> {
  let arguments = (ins F32:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}


def KernelFuncOp : Mock_Op<"kernel", [
  AffineScope, AutomaticAllocationScope, CallableOpInterface,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    static KernelFuncOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         llvm::ArrayRef<mlir::NamedAttribute> attrs = {});
    static KernelFuncOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         mlir::Operation::dialect_attr_range attrs);
    static KernelFuncOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         llvm::ArrayRef<mlir::NamedAttribute> attrs,
                         llvm::ArrayRef<mlir::DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    KernelFuncOp clone(mlir::BlockAndValueMapping &mapper);
    KernelFuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(KernelFuncOp dest, mlir::BlockAndValueMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    llvm::ArrayRef<mlir::Type> getCallableResults() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    llvm::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static mlir::StringRef getDefaultDialect() { return "func"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}


def KernelCallOp : Mock_Op<"kernel_call", [
  CallOpInterface, MemRefsNormalizable, AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Index>:$grid_dim, Variadic<AnyType>:$targets, Variadic<AnyType>:$arguments);

  let extraClassDeclaration = [{
    mlir::FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    mlir::TypeRange getTargetTypes();
  }];

  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $callee `<` $grid_dim `>` `(` $arguments `->` $targets `)` attr-dict `:` functional-type($arguments, $targets)
  }];
}


def KernelReturnOp : Mock_Op<"kernel_return", [
  ReturnLike, Terminator, MemRefsNormalizable, NoSideEffect, HasParent<"KernelFuncOp">
]> {
  let arguments = (ins Variadic<AnyType>:$arguments);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, llvm::None);
  }]>];
}