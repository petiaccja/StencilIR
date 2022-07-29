include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Mock_Dialect : Dialect {
    let name = "mock";
    let cppNamespace = "::mock";
    let hasConstantMaterializer = 0;
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