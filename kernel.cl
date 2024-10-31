
#define BLOCK_SIZE 16
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

inline int _paired(char a, char b) {
    if ((a == 'A' && b == 'U') || (a == 'U' && b == 'A') || (a == 'C' && b == 'G') || (a == 'G' && b == 'C')) {
        return 1;
    }
    return 0;
}


__kernel void myKernel(__global int* B, int N, __global char* seqq, int c0) {


    int c1 = get_group_id(0) + c0;
    int bb = BLOCK_SIZE;

    // Local shared memory
    __local int C[BLOCK_SIZE][BLOCK_SIZE];
    __local int A_elements[BLOCK_SIZE][BLOCK_SIZE];
    __local int B_elements[BLOCK_SIZE][BLOCK_SIZE];
    // Bounds check similar to the original CUDA
    if (c1 <= min((N - 1) / bb, (N + c0 - 2) / bb)) {
        int _sj = c1 - c0;
        int _si = c1;


        for (int m = _sj + 1; m < _si; ++m) {
            // Thread row and column in block
            int row = get_local_id(0);
            int col = get_local_id(1);

            // Assign elements to local memory arrays

        A_elements[row][col] = B[(BLOCK_SIZE * _sj + row) * N + (BLOCK_SIZE * m - 1) + col];
        B_elements[row][col] = B[(BLOCK_SIZE * m + row) * N + (BLOCK_SIZE * _si) + col];


            if (row < BLOCK_SIZE && col < BLOCK_SIZE) {
                int Cvalue = 0;

                // Synchronize all threads in work-group
                barrier(CLK_LOCAL_MEM_FENCE);

                // Unrolled loop for matrix multiplication
                #pragma unroll
                for (int e = 0; e < BLOCK_SIZE; e++) {
                    Cvalue = max(A_elements[row][e] + B_elements[e][col], Cvalue);
                }

                // Synchronize again to ensure correct writing
                barrier(CLK_LOCAL_MEM_FENCE);

                C[row][col] = max(C[row][col], Cvalue);
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
       if (get_local_id(1) == 0) {
        // Further nested loops for matrix and block-level operations
        for (int c2 = max(1, bb * c0 - bb - 1);
             c2 <= min(bb * c0 + bb - 1, N + bb * c0 - bb * c1 - 1); c2 += 1) {

            if (c0 >= 1) {

                int lb = max(bb * c1, -bb * c0 + bb * c1 + c2);
                int ub = min(min(N - 1, bb * c1 + bb - 1), -bb * c0 + bb * c1 + c2 + bb - 1);
                int c3 = get_local_id(0) + lb;

                if (c3 <= ub) {
                    int z = B[(-c2 + c3) * N + c3];


                    int _j = (-c2 + c3) % BLOCK_SIZE;
                    int _i = c3 % BLOCK_SIZE;

                    for (int c4 = 0; c4 < bb - 1; c4 += 1) {
                        z = max(B[(-c2 + c3) * N + (-c2 + c3 + c4)] + B[(-c2 + c3 + c4 + 1) * N + c3], z);
                    }

                    z = max(z, C[_j][_i]);
                    int fragment = (c1 == N / BLOCK_SIZE - 1);

                    for (int c4 = c2 - bb - fragment; c4 < c2; c4 += 1) {
                        z = max(B[(-c2 + c3) * N + (-c2 + c3 + c4)] + B[(-c2 + c3 + c4 + 1) * N + c3], z);
                    }

                    B[(-c2 + c3) * N + c3] = max(z,
                                                 B[(-c2 + c3 + 1) * N + (c3 - 1)] + _paired(seqq[-c2 + c3], seqq[c3]));
                }

            } else {
                int lb = bb * c1 + c2;
                int ub = min(N - 1, bb * c1 + bb - 1);
                int c3 = get_local_id(0) + lb;

                if (c3 <= ub) {
                    int z = B[(-c2 + c3) * N + c3];

                    for (int c4 = 0; c4 < c2; c4 += 1) {
                        z = max(B[(-c2 + c3) * N + (-c2 + c3 + c4)] + B[(-c2 + c3 + c4 + 1) * N + c3], z);
                    }

                    B[(-c2 + c3) * N + c3] = max(z,
                                                 B[(-c2 + c3 + 1) * N + (c3 - 1)] + _paired(seqq[-c2 + c3], seqq[c3]));
                }
            }

        }
        }
    }
}
