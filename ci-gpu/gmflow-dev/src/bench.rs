// Work branch only: the arithmetic peak of the GPU, to compare the matmuls of the model with (see tests/native.rs,
// matmul_ceiling). Same backend-extension pattern as kernels.rs.

use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::{ Dispatch, backend_extension, tensor::FloatTensor };
use burn::tensor::Tensor as BurnTensor;
use burn_cubecl::{ CubeBackend, tensor::CubeTensor };
use cubecl::{ calculate_cube_count_elemwise, cube, prelude::*, CubeDim };

/// 8 independent multiply-add chains of `iters` steps per thread (iters a multiple of 8): 16 * iters flops per thread
#[cube(launch)]
fn fma_peak_kernel<F: Float>(out: &mut Tensor<F>, #[comptime] iters: usize, #[define(F)] _dtype: ElemType) {
    let idx = ABSOLUTE_POS as usize;
    if idx >= out.shape(0) {
        terminate!();
    }
    let x = F::cast_from(idx as u32) * F::new(1e-9_f32);
    let m = F::new(0.999_999_f32);
    let c = F::new(1e-7_f32);
    let mut a0 = x;
    let mut a1 = x + F::new(0.1_f32);
    let mut a2 = x + F::new(0.2_f32);
    let mut a3 = x + F::new(0.3_f32);
    let mut a4 = x + F::new(0.4_f32);
    let mut a5 = x + F::new(0.5_f32);
    let mut a6 = x + F::new(0.6_f32);
    let mut a7 = x + F::new(0.7_f32);
    // 8 steps of the 8 chains per loop iteration, so that the loop itself costs little
    let outer = comptime!(iters / 8);
    for _ in 0..outer {
        #[unroll]
        for _ in 0..8 {
            a0 = fma(a0, m, c);
            a1 = fma(a1, m, c);
            a2 = fma(a2, m, c);
            a3 = fma(a3, m, c);
            a4 = fma(a4, m, c);
            a5 = fma(a5, m, c);
            a6 = fma(a6, m, c);
            a7 = fma(a7, m, c);
        }
    }
    out[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

pub const FMA_ITERS: usize = 4096;

fn launch_fma_peak(like: CubeTensor) -> CubeTensor {
    let n = like.meta.shape().num_elements();
    let dtype = like.dtype;
    let buffer = like.client.empty(n * dtype.size());
    let out = CubeTensor::new_contiguous(like.client.clone(), like.device.clone(), like.meta.shape().clone(), buffer, dtype);
    let dim = CubeDim::new(&like.client, n);
    let count = calculate_cube_count_elemwise(&like.client, n, dim);
    fma_peak_kernel::launch(&out.client, count, dim, out.clone().into_tensor_arg(), FMA_ITERS, dtype_to_storage_type(dtype));
    out
}

#[backend_extension(Cube, Fusion)]
pub trait BenchKernels: burn::backend::Backend {
    #[fusion(dtype = like, shape = like)]
    fn fma_peak(like: FloatTensor<Self>) -> FloatTensor<Self>;
}

impl BenchKernels for CubeBackend {
    fn fma_peak(like: FloatTensor<Self>) -> FloatTensor<Self> {
        launch_fma_peak(like)
    }
}

/// [N] -> [N] after 16 * FMA_ITERS flops per element
pub fn fma_peak(like: BurnTensor<1>) -> BurnTensor<1> {
    BurnTensor::from_dispatch(Dispatch::fma_peak(like.into_dispatch()))
}
