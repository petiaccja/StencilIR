include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"


def Stencil_Dialect : Dialect {
  let name = "stencil";
  let cppNamespace = "::stencil";
  let hasConstantMaterializer = 0;
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class Stencil_Op<string mnemonic, list<Trait> traits = []> :
    Op<Stencil_Dialect, mnemonic, traits>;


def KernelOp : Stencil_Op<"kernel", [
  AffineScope, AutomaticAllocationScope, CallableOpInterface,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    IndexAttr:$num_dimensions
  );
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    static KernelOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         llvm::ArrayRef<mlir::NamedAttribute> attrs = {});
    static KernelOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         mlir::Operation::dialect_attr_range attrs);
    static KernelOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         llvm::ArrayRef<mlir::NamedAttribute> attrs,
                         llvm::ArrayRef<mlir::DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    KernelOp clone(mlir::BlockAndValueMapping &mapper);
    KernelOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(KernelOp dest, mlir::BlockAndValueMapping &mapper);

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


def LaunchKernelOp : Stencil_Op<"launch_kernel", [
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
    $callee `<``<``<` $grid_dim `>``>``>` `(` `(` $arguments `)` `->` `(` $targets `)` `)` attr-dict `:` functional-type($arguments, $targets)
  }];
}


def ReturnOp : Stencil_Op<"return", [
  ReturnLike, Terminator, MemRefsNormalizable, NoSideEffect, HasParent<"KernelOp">
]> {
  let arguments = (ins Variadic<AnyType>:$arguments);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, llvm::None);
  }]>];
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict ($arguments^ `:` type($arguments))?";
}


def IndexOp : Stencil_Op<"index", [NoSideEffect, HasParent<"KernelOp">]> {
  let results = (outs AnyMemRef:$index);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict `:` type($index)";
}


def JumpOp : Stencil_Op<"jump", [NoSideEffect, HasParent<"KernelOp">]> {
  let arguments = (ins AnyMemRef:$input_index, I64ArrayAttr:$offset);
  let results = (outs AnyMemRef:$offseted_index);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$input_index `,` $offset attr-dict `:` functional-type($input_index, $offseted_index)";
}


def SampleOp : Stencil_Op<"sample", [NoSideEffect, HasParent<"KernelOp">]> {
  let arguments = (ins AnyMemRef:$field, AnyMemRef:$index);
  let results = (outs AnyType:$sampled_value);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$field`[` $index `]` attr-dict `:` functional-type(operands, $sampled_value)";
}


def JumpIndirectOp : Stencil_Op<"jump_indirect", [NoSideEffect, HasParent<"KernelOp">]> {
  let arguments = (ins AnyMemRef:$input_index, IndexAttr:$dimension, AnyMemRef:$map, Index:$map_element);
  let results = (outs AnyMemRef:$offseted_index);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$input_index `[` $dimension `]``,` $map `[` $map_element `]` attr-dict `:` functional-type(operands, $offseted_index)";
}


def SampleIndirectOp : Stencil_Op<"sample_indirect", [NoSideEffect, HasParent<"KernelOp">]> {
  let arguments = (ins AnyMemRef:$index, IndexAttr:$dimension, AnyMemRef:$field, Index:$field_element);
  let results = (outs AnyType:$field_value);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$field`[` $index`[`$dimension`]``,` $field_element `]` attr-dict `:` functional-type(operands, $field_value)";
}


def PrintOp : Stencil_Op<"print"> {
  let arguments = (ins F32:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}