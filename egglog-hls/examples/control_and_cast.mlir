// Exercises the constructs added for real frontend output: unary cast
// (index_cast), unary arith (negf), boolean constant, and scf.if with else.
module {
  func.func @mix(%arg0: memref<8xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %flag = arith.constant true
    scf.for %i = %c0 to %n step %c1 {
      %0 = arith.index_cast %i : index to i32
      %1 = memref.load %arg0[%i] : memref<8xf32>
      %2 = arith.negf %1 : f32
      %3 = arith.cmpi eq, %0, %0 : i32
      scf.if %3 {
        memref.store %2, %arg0[%i] : memref<8xf32>
      } else {
        memref.store %1, %arg0[%i] : memref<8xf32>
      }
    }
    return
  }
}
