module {
  func.func @rj_xtime(%arg0: i8) -> i8 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c27_i32 = arith.constant 27 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.extsi %arg0 : i8 to i32
    %1 = arith.andi %0, %c128_i32 : i32
    %2 = arith.cmpi ne, %1, %c0_i32 : i32
    %3 = scf.if %2 -> (i32) {
      %5 = arith.shli %0, %c1_i32 : i32
      %6 = arith.xori %5, %c27_i32 : i32
      scf.yield %6 : i32
    } else {
      %5 = arith.shli %0, %c1_i32 : i32
      scf.yield %5 : i32
    }
    %4 = arith.trunci %3 : i32 to i8
    return %4 : i8
  }
}
}
