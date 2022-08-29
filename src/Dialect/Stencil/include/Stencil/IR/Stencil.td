include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"


def Stencil_Dialect : Dialect {
  let name = "stencil";
  let cppNamespace = "::stencil";
  let hasConstantMaterializer = 0;
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class Stencil_Op<string mnemonic, list<Trait> traits = []> :
    Op<Stencil_Dialect, mnemonic, traits>;


def StencilOp : Stencil_Op<"stencil", [
  AffineScope, AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    IndexAttr:$num_dimensions
  );
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    static StencilOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         llvm::ArrayRef<mlir::NamedAttribute> attrs = {});
    static StencilOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         mlir::Operation::dialect_attr_range attrs);
    static StencilOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         llvm::ArrayRef<mlir::NamedAttribute> attrs,
                         llvm::ArrayRef<mlir::DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    StencilOp clone(mlir::BlockAndValueMapping &mapper);
    StencilOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(StencilOp dest, mlir::BlockAndValueMapping &mapper);

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
    static mlir::StringRef getDefaultDialect() { return "stencil"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let summary = "A compute kernel similar to a CUDA or shader function.";
  let description = [{
    Represents the entry point of a thread in a multi-dimensional grid of threads.
    A kernel is meant for executing stencil operations. 
    A kernel can access its index in the grid, and use that index to sample buffers.
    The kernel can also manipulate the index and perform arithmetic to compute
    zero or more return values. The return values are written into buffers
    at the grid index of the kernel.
  }];
}


def ApplyOp : Stencil_Op<"apply", [
  MemRefsNormalizable,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let arguments = (ins FlatSymbolRefAttr:$callee,
                   Variadic<AnyType>:$inputs,
                   Variadic<AnyShaped>:$outputs,
                   Variadic<Index>:$offsets,
                   I64ArrayAttr:$static_offsets);
  let results = (outs Variadic<AnyShaped>:$results);

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$callee, "::mlir::ValueRange":$inputs, "::mlir::ValueRange":$outputs)>,
    OpBuilder<(ins "::llvm::StringRef":$callee, "::mlir::ValueRange":$inputs, "::mlir::ValueRange":$outputs, "::llvm::ArrayRef<int64_t>":$static_offsets)>,
    OpBuilder<(ins "::llvm::StringRef":$callee, "::mlir::ValueRange":$inputs, "::mlir::ValueRange":$outputs, "::mlir::ValueRange":$offsets)>,
    OpBuilder<(ins "::llvm::StringRef":$callee, "::mlir::ValueRange":$inputs, "::mlir::ValueRange":$outputs, "::mlir::ValueRange":$offsets, "::llvm::ArrayRef<int64_t>":$static_offsets)>
  ];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $callee `(` $inputs `)` `->` $outputs `,` `offsets` `` `:` custom<OperandsOrIntegersSizesList>($offsets, $static_offsets) attr-dict `:` functional-type(operands, $results)
  }];
  let summary = "Launch a kernel on a multi-dimensional grid, like CUDA or compute shaders.";
  let description = [{
    For each index in the grid, a thread is launched that starts executing the kernel.
    The index is passed to the kernel. The values returned from the kernel are written
    into the targets of this operation at the threads's index.

    In theory, all threads are executed in parallel, through in practice this reduces
    to a parallel OpenMP loop or a GPU kernel call.
  }];
}


def InvokeStencilOp : Stencil_Op<"invoke_stencil", [
  MemRefsNormalizable, AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Index>:$indices, Variadic<AnyType>:$arguments);
  let results = (outs Variadic<AnyType>);
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
  }];

  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $callee `<` $indices `>` `(` $arguments `)` attr-dict `:` functional-type($arguments, results)
  }];
}


def ReturnOp : Stencil_Op<"return", [
  ReturnLike, Terminator, MemRefsNormalizable, NoSideEffect, HasParent<"StencilOp">
]> {
  let arguments = (ins Variadic<AnyType>:$arguments);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, llvm::None);
  }]>];
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict ($arguments^ `:` type($arguments))?";
  let summary = "Return values from a kernel.";
}


def IndexOp : Stencil_Op<"index", [NoSideEffect, HasParent<"StencilOp">]> {
  let results = (outs AnyVector:$index);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict `:` type($index)";
  let summary = "Get the index of this kernel in the grid.";
}


def JumpOp : Stencil_Op<"jump", [NoSideEffect]> {
  let arguments = (ins AnyVector:$input_index, I64ArrayAttr:$offset);
  let results = (outs AnyVector:$offseted_index);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$input_index `,` $offset attr-dict `:` functional-type($input_index, $offseted_index)";
  let summary = "Move an index by a constant offset.";
}


def SampleOp : Stencil_Op<"sample", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyShaped:$field, AnyVector:$index);
  let results = (outs AnyType:$sampled_value);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$field`[` $index `]` attr-dict `:` functional-type(operands, $sampled_value)";
  let summary = "Get the element from a field at the specified index.";
}


def JumpIndirectOp : Stencil_Op<"jump_indirect", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyVector:$input_index, IndexAttr:$dimension, AnyShaped:$map, Index:$map_element);
  let results = (outs AnyVector:$offseted_index);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$input_index `[` $dimension `]``,` $map `[` $map_element `]` attr-dict `:` functional-type(operands, $offseted_index)";
  let summary = "Use a connectivity map to modify the index.";
  let description = [{
    The procedure is as follows:
    1. index_element = input_index[dimension]
    2. new_element = map[index_element, map_element]
    3. input_index[dimension] = new_element

    Essentially, one element of the index is replaced by a value from the map.
    The modified index is returned, while the input index is in fact left unchanged.
  }];
}


def SampleIndirectOp : Stencil_Op<"sample_indirect", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyVector:$index, IndexAttr:$dimension, AnyShaped:$field, Index:$field_element);
  let results = (outs AnyType:$field_value);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$field`[` $index`[`$dimension`]``,` $field_element `]` attr-dict `:` functional-type(operands, $field_value)";
  let summary = "Sample a field of the same shape as a connectivity map.";
  let description = [{
    The procedure is as follows:
    1. index_element = input_index[dimension]
    2. sample = field[index_element, map_element]

    The sample is then returned by this operation.
  }];
}


def PrintOp : Stencil_Op<"print"> {
  let arguments = (ins F32:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let summary = "Prints a float32 to STDOUT.";
  let description = [{
    It's best not to use use this at all, except for debugging on the CPU.
  }];
}