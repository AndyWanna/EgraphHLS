// Illustrative fixture (hand-written, NOT yet emitted by the real front-end).
// "Regular" form of matmul_store: nested scf.for with a trailing memref.store.
// Regenerate from the chosen HLS->MLIR front-end (ScaleHLS) once it is stood up.
//
// for i: for j: R[i][j] = 0; for k: R[i][j] += A[i][k]*B[k][j]

func.func @matmul_store(%A: memref<32x32xi32>, %B: memref<32x32xi32>, %R: memref<32x32xi32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c32 step %c1 {
      memref.store %zero, %R[%i, %j] : memref<32x32xi32>
      scf.for %k = %c0 to %c32 step %c1 {
        %a = memref.load %A[%i, %k] : memref<32x32xi32>
        %b = memref.load %B[%k, %j] : memref<32x32xi32>
        %acc = memref.load %R[%i, %j] : memref<32x32xi32>
        %prod = arith.muli %a, %b : i32
        %sum = arith.addi %acc, %prod : i32
        memref.store %sum, %R[%i, %j] : memref<32x32xi32>
      }
    }
  }
  return
}
