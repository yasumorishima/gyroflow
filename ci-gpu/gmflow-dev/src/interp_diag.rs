// Work branch only (never part of #1143): burn's bilinear interpolation with a given cubek strategy instead of the one
// autotune picks, to find whether one of the two is wrong for a batch of more than one. The bidirectional check failed
// now and then with the same wrong flow on MSL and WGSL, in a run whose autotune log shows minimize_latency picked for
// the flow doubled at 1/8 (tuned at batch 1: the interpolation key has no batch), where 18 passing runs picked
// maximize_throughput
use burn::backend::{ Dispatch, backend_extension, tensor::FloatTensor };
use burn::tensor::ops::{ InterpolateMode, InterpolateOptions };
use burn::tensor::{ Shape, Tensor };
use burn_cubecl::CubeBackend;
use cubek::interpolate::InterpolateStrategy;

fn up2_shape(x: &Shape) -> Shape {
    Shape::new([x[0], x[1], x[2] * 2, x[3] * 2])
}

#[backend_extension(Cube, Fusion)]
pub trait InterpDiag: burn::backend::Backend {
    #[fusion(dtype = x, shape = up2_shape(x))]
    fn interp2x(x: FloatTensor<Self>, latency: bool) -> FloatTensor<Self>;
}

impl InterpDiag for CubeBackend {
    fn interp2x(x: FloatTensor<Self>, latency: bool) -> FloatTensor<Self> {
        let s = x.meta.shape().clone();
        let strategy = if latency { InterpolateStrategy::MinimizeLatency } else { InterpolateStrategy::MaximizeThroughput };
        let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(true);
        burn_cubecl::kernel::interpolate::execute_interpolate(x, [s[2] * 2, s[3] * 2], options, strategy).expect("interpolate")
    }
}

/// x [B, C, H, W] -> [B, C, 2H, 2W], bilinear with aligned corners, by the given strategy
pub fn interp2x(x: Tensor<4>, latency: bool) -> Tensor<4> {
    Tensor::from_dispatch(Dispatch::interp2x(x.into_dispatch(), latency))
}
