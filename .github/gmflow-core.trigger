gyroflow-core with the hand-written gmflow (2026-10-22)
rerun after the concat1 fix (2026-10-22)
rerun with the minimal lock, burn 0.22.0 (2026-10-22)
rerun after the device closure fix (2026-10-22)
rerun with burn 0.22.1 (fusion binding limit fix) and gpu-allocator on windows 0.62 (2026-10-22)
1d088a7 fused qkv removed
8651318 gmflow: stage times by a device sync at each point (a read of one value did not wait for the stage), and the final read counted core rerun (macOS runners were not acquired)
91aee51 gmflow: corr_with_flow from the products with the integer window of each position, then the bilinear weighting (100 products of C per position instead of 324); integer flows in the kernel check
e4259eb gmflow: correlation taps one window row per thread (independent sums); convolutions chosen from the f32 matrix units of the device instead of a timing at load; matmul ceiling test (work branch)
ab5d253 gmflow: own register-tiled matmul (kernels::gemm) for the im2col convolutions and the GRU where the device has no f32 matrix units (burn: 8-16% of the arithmetic peak on WGSL)
40fe539 gmflow: gemm kernel with usize literals (an untyped 0 in a branch was taken as i32 by the cube macro)
549fc5f gmflow: gemm with 128 x 128 blocks, 8 x 8 outputs a thread, k step 8 (4 multiply-adds per value read from shared memory), rows of a padded to 132; burn for convolutions of under 32 outputs
gemm tilings and gemm_linear 2026-10-22T23:42:48Z
gemm 64/8/4 and linear layers 2026-10-23T02:22:27Z
flow_double 2026-10-23T09:20:13Z
round 4b: fused kernels on by default
round 4c: gemm4 as the default gemm
round 5: implicit gemm (Ops.implicit, off in production)
implicit GEMM convolutions on Apple
round 7: cooperative matrices (Ops.coop) where WGSL has them
round 7b: WGSL kernels on every Mac (no native MSL)
round 7b: the lock of src/core too
round 7c: audit fixes (window guard order, coop_works window attention)
round 8: fused LayerNorm (+ residual) in the transformer
round 9: fused GRU gates
round 10: fused InstanceNorm (+ relu, residual) in the CNN
round 11: the load timed apart from the first pass (weights, coop check)
round 12: coop_works against CPU references (no burn autotune at load)
round 13: softmax_flow in global matching and propagation; a second start measured
round 14: a passed coop_works kept per GPU, driver and kernel sources
round 15: GELU in fc2's loads, conv bias + relu as coop writes, the FFN's cat read by fc1
