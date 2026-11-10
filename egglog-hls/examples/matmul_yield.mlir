// "Annoying form": the accumulator is a loop-carried value via scf.for
// iter_args + scf.yield. Semantically identical to matmul_regular, but the
// irregular shape breaks many ScaleHLS passes. reg2mem normalizes it back
// to the store form; see egglog_hls.normalize.
module {
  func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        %init = memref.load %C[%i, %j] : memref<4x4xf32>
        %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (f32) {
          %0 = memref.load %A[%i, %k] : memref<4x4xf32>
          %1 = memref.load %B[%k, %j] : memref<4x4xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = arith.addf %acc, %2 : f32
          scf.yield %3 : f32
        }
        memref.store %sum, %C[%i, %j] : memref<4x4xf32>
      }
    }
    return
  }
}
