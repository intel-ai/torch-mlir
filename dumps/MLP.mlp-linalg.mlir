#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "MLP"} {
  func.func @MLP(%arg0: tensor<128x262144xf32>, %arg1: tensor<128xf32>, %arg2: tensor<64x512x512xf32>) -> tensor<64x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg2 [[0], [1, 2]] : tensor<64x512x512xf32> into tensor<64x262144xf32>
    %0 = tensor.empty() : tensor<262144x128xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<128x262144xf32>) outs(%0 : tensor<262144x128xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<64x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %3 = linalg.matmul ins(%collapsed, %transposed : tensor<64x262144xf32>, tensor<262144x128xf32>) outs(%2 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %3 : tensor<128xf32>, tensor<64x128xf32>) outs(%1 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<64x128xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<64x128xf32>) outs(%1 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.cmpf ugt, %in, %cst : f32
      %7 = arith.select %6, %in, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<64x128xf32>
    return %5 : tensor<64x128xf32>
  }
}

