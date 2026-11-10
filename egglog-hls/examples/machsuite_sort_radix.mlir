module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi32>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu", "polygeist.target-cpu" = "generic", "polygeist.target-features" = "+neon,+outline-atomics,+v8a"} {
  func.func @ss_sort(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg4 = %c0 to %c32 step %c2 iter_args(%arg5 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg4 : index to i32
      func.call @init(%arg2) : (memref<?xi32>) -> ()
      %2 = arith.cmpi eq, %arg5, %c0_i32 : i32
      scf.if %2 {
        func.call @hist(%arg2, %arg0, %1) : (memref<?xi32>, memref<?xi32>, i32) -> ()
      } else {
        func.call @hist(%arg2, %arg1, %1) : (memref<?xi32>, memref<?xi32>, i32) -> ()
      }
      func.call @local_scan(%arg2) : (memref<?xi32>) -> ()
      func.call @sum_scan(%arg3, %arg2) : (memref<?xi32>, memref<?xi32>) -> ()
      func.call @last_step_scan(%arg2, %arg3) : (memref<?xi32>, memref<?xi32>) -> ()
      %3 = arith.extui %2 : i1 to i32
      scf.if %2 {
        func.call @update(%arg1, %arg2, %arg0, %1) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> ()
      } else {
        func.call @update(%arg0, %arg2, %arg1, %1) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> ()
      }
      scf.yield %3 : i32
    }
    return
  }
  func.func @init(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg1 = %c0 to %c2048 step %c1 {
      memref.store %c0_i32, %arg0[%arg1] : memref<?xi32>
    }
    return
  }
  func.func @hist(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c512_i32 = arith.constant 512 : i32
    scf.for %arg3 = %c0 to %c512 step %c1 {
      %0 = arith.index_cast %arg3 : index to i32
      scf.for %arg4 = %c0 to %c4 step %c1 {
        %1 = arith.index_cast %arg4 : index to i32
        %2 = arith.muli %0, %c4_i32 : i32
        %3 = arith.addi %2, %1 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg1[%4] : memref<?xi32>
        %6 = arith.shrsi %5, %arg2 : i32
        %7 = arith.andi %6, %c3_i32 : i32
        %8 = arith.muli %7, %c512_i32 : i32
        %9 = arith.addi %8, %0 : i32
        %10 = arith.addi %9, %c1_i32 : i32
        %11 = arith.index_cast %10 : i32 to index
        %12 = memref.load %arg0[%11] : memref<?xi32>
        %13 = arith.addi %12, %c1_i32 : i32
        memref.store %13, %arg0[%11] : memref<?xi32>
      }
    }
    return
  }
  func.func @local_scan(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i32 = arith.constant 16 : i32
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %0 = arith.index_cast %arg1 : index to i32
      scf.for %arg2 = %c1 to %c16 step %c1 {
        %1 = arith.index_cast %arg2 : index to i32
        %2 = arith.muli %0, %c16_i32 : i32
        %3 = arith.addi %2, %1 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = arith.addi %3, %c-1_i32 : i32
        %6 = arith.index_cast %5 : i32 to index
        %7 = memref.load %arg0[%6] : memref<?xi32>
        %8 = memref.load %arg0[%4] : memref<?xi32>
        %9 = arith.addi %8, %7 : i32
        memref.store %9, %arg0[%4] : memref<?xi32>
      }
    }
    return
  }
  func.func @sum_scan(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.store %c0_i32, %arg0[0] : memref<?xi32>
    scf.for %arg2 = %c1 to %c128 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c16_i32 : i32
      %2 = arith.addi %1, %c-1_i32 : i32
      %3 = arith.addi %0, %c-1_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = memref.load %arg0[%4] : memref<?xi32>
      %6 = arith.index_cast %2 : i32 to index
      %7 = memref.load %arg1[%6] : memref<?xi32>
      %8 = arith.addi %5, %7 : i32
      memref.store %8, %arg0[%arg2] : memref<?xi32>
    }
    return
  }
  func.func @last_step_scan(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i32 = arith.constant 16 : i32
    scf.for %arg2 = %c0 to %c128 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %1 = arith.index_cast %arg3 : index to i32
        %2 = arith.muli %0, %c16_i32 : i32
        %3 = arith.addi %2, %1 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg0[%4] : memref<?xi32>
        %6 = memref.load %arg1[%arg2] : memref<?xi32>
        %7 = arith.addi %5, %6 : i32
        memref.store %7, %arg0[%4] : memref<?xi32>
      }
    }
    return
  }
  func.func @update(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c512_i32 = arith.constant 512 : i32
    scf.for %arg4 = %c0 to %c512 step %c1 {
      %0 = arith.index_cast %arg4 : index to i32
      scf.for %arg5 = %c0 to %c4 step %c1 {
        %1 = arith.index_cast %arg5 : index to i32
        %2 = arith.muli %0, %c4_i32 : i32
        %3 = arith.addi %2, %1 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg2[%4] : memref<?xi32>
        %6 = arith.shrsi %5, %arg3 : i32
        %7 = arith.andi %6, %c3_i32 : i32
        %8 = arith.muli %7, %c512_i32 : i32
        %9 = arith.addi %8, %0 : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = memref.load %arg1[%10] : memref<?xi32>
        %12 = arith.index_cast %11 : i32 to index
        memref.store %5, %arg0[%12] : memref<?xi32>
        %13 = memref.load %arg1[%10] : memref<?xi32>
        %14 = arith.addi %13, %c1_i32 : i32
        memref.store %14, %arg1[%10] : memref<?xi32>
      }
    }
    return
  }
}
