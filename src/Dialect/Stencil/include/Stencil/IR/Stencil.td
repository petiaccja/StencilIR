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
  let useFoldAPI = kEmitFoldAdaptorFolder; 
}

class Stencil_Op<string mnemonic, list<Trait> traits = []> :
    Op<Stencil_Dialect, mnemonic, traits>;


def StencilOp : Stencil_Op<"stencil", [
  AffineScope, AutomaticAllocationScope, CallableOpInterface,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    I64Attr:$num_dimensions,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
  let builders = [OpBuilder<(ins
    "::llvm::StringRef":$name,
    "::mlir::FunctionType":$type,
    "::mlir::IntegerAttr":$num_dimensions,
    CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs,
    CArg<"::llvm::ArrayRef<::mlir::DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    static StencilOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         ::mlir::IntegerAttr numDimensions, llvm::ArrayRef<mlir::NamedAttribute> attrs = {});
    static StencilOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         ::mlir::IntegerAttr numDimensions, mlir::Operation::dialect_attr_range attrs);
    static StencilOp create(mlir::Location location, mlir::StringRef name, mlir::FunctionType type,
                         ::mlir::IntegerAttr numDimensions, 
                         llvm::ArrayRef<mlir::NamedAttribute> attrs,
                         llvm::ArrayRef<mlir::DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    StencilOp clone(mlir::IRMapping &mapper);
    StencilOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(StencilOp dest, mlir::IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ::mlir::ArrayRef<::mlir::Type> getCallableResults() { return getFunctionType().getResults(); }

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
    $callee `(` $inputs `)` `outs` `(` $outputs `)` `offsets` $offsets $static_offsets attr-dict `:` functional-type(operands, $results)
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


def GenerateOp : Stencil_Op<"generate", [
  MemRefsNormalizable,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins Variadic<AnyShaped>:$outputs,
                   Variadic<Index>:$offsets,
                   I64ArrayAttr:$static_offsets);
  let results = (outs Variadic<AnyShaped>:$results);
  let regions = (region AnyRegion:$body);
  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$outputs)>,
    OpBuilder<(ins "::mlir::ValueRange":$outputs, "::llvm::ArrayRef<int64_t>":$static_offsets)>,
    OpBuilder<(ins "::mlir::ValueRange":$outputs, "::mlir::ValueRange":$offsets)>,
    OpBuilder<(ins "::mlir::ValueRange":$outputs, "::mlir::ValueRange":$offsets, "::llvm::ArrayRef<int64_t>":$static_offsets)>
  ];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    ` ` `outs` `(` $outputs `)` `offsets` $offsets `` $static_offsets attr-dict `:` functional-type(operands, $results) $body
  }];
  let summary = "TODO";
  let description = [{
    TODO
  }];
  let extraClassDeclaration = [{
      ::mlir::Block *addEntryBlock();
  }];  
}


def YieldOp : Stencil_Op<"yield", [
  ReturnLike, Terminator, MemRefsNormalizable, Pure, HasParent<"GenerateOp">
]> {
  let arguments = (ins Variadic<AnyType>:$arguments);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict ($arguments^ `:` type($arguments))?";
  let summary = "TODO.";
}


def InvokeOp : Stencil_Op<"invoke", [
  MemRefsNormalizable, CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let arguments = (ins FlatSymbolRefAttr:$callee, AnyVector:$index, Variadic<AnyType>:$arguments);
  let results = (outs Variadic<AnyType>);
  let extraClassDeclaration = [{
    mlir::FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin() + 1; }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $callee `<` $index `>` `` `(` $arguments `)` attr-dict `:` functional-type(operands, results)
  }];
}


def ReturnOp : Stencil_Op<"return", [
  ReturnLike, Terminator, MemRefsNormalizable, Pure, HasParent<"StencilOp">
]> {
  let arguments = (ins Variadic<AnyType>:$arguments);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict ($arguments^ `:` type($arguments))?";
  let summary = "Return values from a kernel.";
}


def IndexOp : Stencil_Op<"index", [NoMemoryEffect]> {
  let results = (outs AnyVector:$index);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($index)";
  let summary = "Get the index of this kernel in the grid.";
}


def JumpOp : Stencil_Op<"jump", [NoMemoryEffect]> {
  let arguments = (ins AnyVector:$input_index, I64ArrayAttr:$offset);
  let results = (outs AnyVector:$offseted_index);
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$input_index `,` $offset attr-dict `:` functional-type($input_index, $offseted_index)";
  let summary = "Move an index by a constant offset.";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}


def ProjectOp : Stencil_Op<"project", [NoMemoryEffect]> {
  let arguments = (ins AnyVector:$source, I64ArrayAttr:$positions);
  let results = (outs AnyVector:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$source, "::mlir::ArrayRef<int64_t>":$positions)>
  ];
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$source `` $positions attr-dict `:` functional-type($source, $result)";
  let summary = "Project and optionally shuffle the elements of an index.";
  let hasFolder = 1;
}


def ExtendOp : Stencil_Op<"extend", [NoMemoryEffect]> {
  let arguments = (ins AnyVector:$source, IndexAttr:$position, Index:$value);
  let results = (outs AnyVector:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$inputIndex, "int64_t":$dimension, "::mlir::Value":$value)>
  ];
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$source `[` $position `]` `,` $value attr-dict `:` functional-type($source, $result)";
  let summary = "Extend the index by inserting an element at position.";
}


def ExchangeOp : Stencil_Op<"exchange", [NoMemoryEffect]> {
  let arguments = (ins AnyVector:$source, IndexAttr:$position, Index:$value);
  let results = (outs AnyVector:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$inputIndex, "int64_t":$position, "::mlir::Value":$value)>
  ];
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$source `[` $position `]` `,` $value attr-dict `:` functional-type($source, $result)";
  let summary = "Exchange of the index at position.";
}


def ExtractOp : Stencil_Op<"extract", [NoMemoryEffect]> {
  let arguments = (ins AnyVector:$source, IndexAttr:$position);
  let results = (outs Index:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$inputIndex, "int64_t":$position)>
  ];
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$source `[` $position `]` attr-dict `:` type($source)";
  let summary = "Exchange of the index at position.";
}


def SampleOp : Stencil_Op<"sample", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyShaped:$field, AnyVector:$index);
  let results = (outs AnyType:$sampled_value);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let assemblyFormat = "$field`[` $index `]` attr-dict `:` functional-type(operands, $sampled_value)";
  let summary = "Get the element from a field at the specified index.";
}


def PrintOp : Stencil_Op<"print"> {
  let arguments = (ins F32:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let summary = "Prints a float32 to STDOUT.";
  let description = [{
    It's best not to use use this at all, except for debugging on the CPU.
  }];
}