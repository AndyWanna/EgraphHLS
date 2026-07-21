module {
  func.func @vecadd(%A: memref<8xf32>, %B: memref<8xf32>, %C: memref<8xf32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %0 = memref.load %A[%i] : memref<8xf32>
      %1 = memref.load %B[%i] : memref<8xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %C[%i] : memref<8xf32>
    }
    return
  }
}
