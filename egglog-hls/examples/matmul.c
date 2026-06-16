// Matmul from the Jun 16 call screenshot. R = A * B, N x N, int32.
// The two C styles below lower to the two different MLIR shapes documented in
// ../README.md (regular store form vs scf.for + yield/iter_args form).

#define N 32

// (a) store-into-array style -> lowers to the "regular" scf.for form
//     (trailing memref.store, easy to optimize).
void matmul_store(int A[N][N], int B[N][N], int R[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            R[i][j] = 0;
            for (int k = 0; k < N; k++) {
                R[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// (b) scalar-accumulator style -> often lowers to the "yield" form
//     (loop-carried `r` via scf.for iter_args + scf.yield).
void matmul_accum(int A[N][N], int B[N][N], int R[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int r = 0;
            for (int k = 0; k < N; k++) {
                r += A[i][k] * B[k][j];
            }
            R[i][j] = r;
        }
    }
}
