// "Clean form": accumulation lowered to an explicit memref.store into C.
// Regular, easy for ScaleHLS passes to consume.
module {
  func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        scf.for %k = %c0 to %c4 step %c1 {
          %0 = memref.load %A[%i, %k] : memref<4x4xf32>
          %1 = memref.load %B[%k, %j] : memref<4x4xf32>
          %2 = memref.load %C[%i, %j] : memref<4x4xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %C[%i, %j] : memref<4x4xf32>
        }
      }
    }
    return
  }
}
