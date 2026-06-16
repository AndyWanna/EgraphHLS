// Illustrative fixture (hand-written, NOT yet emitted by the real front-end).
// "Yield" form of matmul_accum: the inner loop carries the accumulator `r` via
// scf.for iter_args and returns it with scf.yield. Semantically identical to
// matmul_regular.mlir but irregular and harder to optimize (the gotcha from the
// Jun 16 call). Normalize via reg2mem -> optimize -> mem2reg.
//
// for i: for j: r=0; for k: r += A[i][k]*B[k][j]; R[i][j] = r

func.func @matmul_accum(%A: memref<32x32xi32>, %B: memref<32x32xi32>, %R: memref<32x32xi32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c32 step %c1 {
      %r = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %zero) -> (i32) {
        %a = memref.load %A[%i, %k] : memref<32x32xi32>
        %b = memref.load %B[%k, %j] : memref<32x32xi32>
        %prod = arith.muli %a, %b : i32
        %sum = arith.addi %acc, %prod : i32
        scf.yield %sum : i32
      }
      memref.store %r, %R[%i, %j] : memref<32x32xi32>
    }
  }
  return
}
