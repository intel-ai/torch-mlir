#ifndef TILING_INTERFACE
#define TILING_INTERFACE

include "mlir/IR/OpBase.td"

def TilingOpInterface : OpInterface<"Tiling"> {
  let description = [{
    Interface to tile.
  }];

  let methods = [
    InterfaceMethod<"Tile operation.",
                    /*retType=*/"void",
                    /*methodName=*/"tile">
  ];
}

#endif // TILING_INTERFACE
