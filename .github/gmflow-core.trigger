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
