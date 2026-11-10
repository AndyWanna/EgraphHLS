module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi32>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu", "polygeist.target-cpu" = "generic", "polygeist.target-features" = "+neon,+outline-atomics,+v8a"} {
  func.func @matmul(%arg0: memref<?x4xf32>, %arg1: memref<?x4xf32>, %arg2: memref<?x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<?x4xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<?x4xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = memref.load %arg2[%arg3, %arg4] : memref<?x4xf32>
          %4 = arith.addf %3, %2 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<?x4xf32>
        }
      }
    }
    return
  }
}
