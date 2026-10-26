// Work branch only (never part of #1143): a matmul and a convolution on the matrix units of Apple GPUs from WGSL. wgpu 30
// on Metal offers cooperative matrices (Features::EXPERIMENTAL_COOPERATIVE_MATRIX, `enable wgpu_cooperative_matrix;`),
// which naga writes as simdgroup_matrix (8 x 8, f32); cubecl's WGSL compiler does not use them (0 cmma configurations on
// WGSL), so the kernels are hand-written WGSL (wgsl.rs) that cubecl launches as is (CubeKernel::source) on burn's buffers.

#![allow(unused_imports)]

use crate::wgsl::{ self, CoopB, CoopShape, CoopTile };
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{ Dispatch, backend_extension, tensor::FloatTensor };
use burn::tensor::{ DType, Shape, Tensor as BurnTensor };
use burn_cubecl::{ CubeBackend, kernel::into_contiguous, tensor::CubeTensor };
use cubecl::ir::{ UIntKind, metadata::Info };
// BufferIOAttr comes with the prelude (cubecl_runtime::kernel::*)
use cubecl::prelude::*;
use cubecl::server::KernelArguments;

/// One compiled kernel per source: the id covers the whole text, which is what the compilation caches are keyed by
struct CoopKernel {
    source: String,
    threads: u32,
}

impl KernelMetadata for CoopKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.source.clone())
    }

    fn address_type(&self) -> ElemType {
        UIntKind::U32.into()
    }
}

impl CubeKernel for CoopKernel {
    /// Not compiled (source() is): only the settings of the launch
    fn define(&self) -> KernelDefinition {
        let settings = KernelSettings::new(CubeDim::new_1d(self.threads).0, ExecutionMode::Checked, AddressType::U32);
        KernelDefinition { body: Scope::root(settings.clone()), settings, info: Info::default() }
    }

    fn source(&self) -> Option<PrecompiledSource> {
        Some(PrecompiledSource { source: self.source.clone(), entrypoint_name: "main".to_string(), lang: "wgsl" })
    }
}

/// c = a @ b on `batch` workgroup layers into a new [batch, m, n]; a and b declared read only to cubecl, so that a
/// kernel that fails to compile marks only its output, not inputs other launches share
fn launch(a: &CubeTensor, b: &CubeTensor, shape: CoopShape, batch: usize, tile: CoopTile) -> CubeTensor {
    assert_eq!(a.dtype, DType::F32, "coop: f32 only");
    let source = wgsl::source(shape, tile);
    let out_shape = Shape::new([batch, shape.m, shape.n]);
    let buffer = a.client.empty(out_shape.num_elements() * a.dtype.size());
    let out = CubeTensor::new_contiguous(a.client.clone(), a.device.clone(), out_shape, buffer, a.dtype);
    let count = CubeCount::Static(shape.n.div_ceil(tile.cols()) as u32, shape.m.div_ceil(tile.rows()) as u32, batch as u32);
    let args = KernelArguments::new()
        .with_buffer_io(a.handle.clone().binding(), BufferIOAttr::ReadOnly)
        .with_buffer_io(b.handle.clone().binding(), BufferIOAttr::ReadOnly)
        .with_buffer_io(out.handle.clone().binding(), BufferIOAttr::WriteOnly);
    out.client.launch(Box::new(CoopKernel { source, threads: tile.threads() as u32 }), count, args);
    out
}

fn coop_shape(a: &Shape, b: &Shape) -> Shape {
    Shape::new([a[0].max(b[0]), a[1], b[2]])
}

fn launch_gemm(a: CubeTensor, b: CubeTensor, tile: CoopTile) -> CubeTensor {
    let (a, b) = (into_contiguous(a), into_contiguous(b));
    let (sa, sb) = (a.meta.shape().clone(), b.meta.shape().clone());
    assert!(sa[2] == sb[1], "coop gemm: inner dimensions {} and {}", sa[2], sb[1]);
    assert!(sa[0] == sb[0] || sa[0] == 1 || sb[0] == 1, "coop gemm: batches {} and {}", sa[0], sb[0]);
    let (m, k, n) = (sa[1], sa[2], sb[2]);
    let shape = CoopShape {
        m,
        n,
        k,
        stride_a: if sa[0] == 1 { 0 } else { m * k },
        b: CoopB::Matrix { stride: if sb[0] == 1 { 0 } else { k * n } },
    };
    launch(&a, &b, shape, sa[0].max(sb[0]), tile)
}

fn conv_shape(w: &Shape, x: &Shape, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize) -> Shape {
    let (oh, ow) = CoopB::conv_out(x[1], x[2], x[3], kh, kw, stride, ph, pw);
    Shape::new([x[0], w[1], oh * ow])
}

#[allow(clippy::too_many_arguments)]
fn launch_conv(w: CubeTensor, x: CubeTensor, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize, tile: CoopTile) -> CubeTensor {
    let (w, x) = (into_contiguous(w), into_contiguous(x));
    let (sw, sx) = (w.meta.shape().clone(), x.meta.shape().clone());
    let (c, h, wd) = (sx[1], sx[2], sx[3]);
    assert!(sw[0] == 1 && sw[2] == c * kh * kw, "coop conv: weight {sw:?} for x {sx:?} and a {kh}x{kw} kernel");
    let (oh, ow) = CoopB::conv_out(c, h, wd, kh, kw, stride, ph, pw);
    let shape = CoopShape { m: sw[1], n: oh * ow, k: sw[2], stride_a: 0, b: CoopB::Conv { c, h, w: wd, kh, kw, stride, ph, pw } };
    launch(&w, &x, shape, sx[0], tile)
}

#[backend_extension(Cube, Fusion)]
pub trait CoopKernels: burn::backend::Backend {
    #[fusion(dtype = a, shape = coop_shape(a, b))]
    fn coop_gemm(a: FloatTensor<Self>, b: FloatTensor<Self>, bk: usize, sgr: usize, sgc: usize, fr: usize, fc: usize) -> FloatTensor<Self>;
    #[fusion(dtype = x, shape = conv_shape(w, x, *kh, *kw, *stride, *ph, *pw))]
    #[allow(clippy::too_many_arguments)]
    fn coop_conv(w: FloatTensor<Self>, x: FloatTensor<Self>, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize, bk: usize, sgr: usize, sgc: usize, fr: usize, fc: usize) -> FloatTensor<Self>;
}

impl CoopKernels for CubeBackend {
    fn coop_gemm(a: FloatTensor<Self>, b: FloatTensor<Self>, bk: usize, sgr: usize, sgc: usize, fr: usize, fc: usize) -> FloatTensor<Self> {
        launch_gemm(a, b, CoopTile { bk, sgr, sgc, fr, fc })
    }
    fn coop_conv(w: FloatTensor<Self>, x: FloatTensor<Self>, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize, bk: usize, sgr: usize, sgc: usize, fr: usize, fc: usize) -> FloatTensor<Self> {
        launch_conv(w, x, kh, kw, stride, ph, pw, CoopTile { bk, sgr, sgc, fr, fc })
    }
}

/// [batch, m, k] @ [batch, k, n] -> [batch, m, n] (an operand of batch 1 shared), f32, any sizes
pub fn gemm(a: BurnTensor<3>, b: BurnTensor<3>, tile: CoopTile) -> BurnTensor<3> {
    let CoopTile { bk, sgr, sgc, fr, fc } = tile;
    BurnTensor::from_dispatch(Dispatch::coop_gemm(a.into_dispatch(), b.into_dispatch(), bk, sgr, sgc, fr, fc))
}

/// w [1, Co, C*KH*KW] @ the KH x KW columns of x [N, C, H, W] (stride, symmetric zero padding) -> [N, Co, OH*OW], the
/// columns gathered in the kernel
#[allow(clippy::too_many_arguments)]
pub fn conv_gemm(w: BurnTensor<3>, x: BurnTensor<4>, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize, tile: CoopTile) -> BurnTensor<3> {
    let CoopTile { bk, sgr, sgc, fr, fc } = tile;
    BurnTensor::from_dispatch(Dispatch::coop_conv(w.into_dispatch(), x.into_dispatch(), kh, kw, stride, ph, pw, bk, sgr, sgc, fr, fc))
}

/// A 2D convolution (symmetric zero padding, no dilation or groups): x [N, C, H, W], weight [Cout, C, KH, KW] ->
/// [N, Cout, OH, OW]; 1x1 of stride 1 as a plain matmul of the weight and x, the others gathered
pub fn conv2d(x: BurnTensor<4>, weight: BurnTensor<4>, bias: Option<BurnTensor<1>>, stride: usize, padding: [usize; 2], tile: CoopTile) -> BurnTensor<4> {
    let [n, c, h, w] = x.dims();
    let [cout, cin, kh, kw] = weight.dims();
    assert_eq!(c, cin, "coop conv2d: channels");
    let (ph, pw) = (padding[0], padding[1]);
    let (oh, ow) = CoopB::conv_out(c, h, w, kh, kw, stride, ph, pw);
    let wm = weight.reshape([1, cout, c * kh * kw]);
    let out = if kh == 1 && kw == 1 && stride == 1 && ph == 0 && pw == 0 {
        gemm(wm, x.reshape([n, c, h * w]), tile)
    } else {
        conv_gemm(wm, x, kh, kw, stride, ph, pw, tile)
    };
    let out = out.reshape([n, cout, oh, ow]);
    match bias {
        Some(b) => out + b.reshape([1, cout, 1, 1]),
        None => out,
    }
}
