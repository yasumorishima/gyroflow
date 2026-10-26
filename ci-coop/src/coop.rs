// Work branch only (never part of #1143): a matmul on the matrix units of Apple GPUs from WGSL. wgpu 30 on Metal offers
// cooperative matrices (Features::EXPERIMENTAL_COOPERATIVE_MATRIX, `enable wgpu_cooperative_matrix;`), which naga
// writes as simdgroup_matrix (8 x 8, f32); cubecl's WGSL compiler does not use them (0 cmma configurations on WGSL), so
// the kernel is hand-written WGSL (wgsl.rs) that cubecl launches as is (CubeKernel::source) on burn's buffers.

#![allow(unused_imports)]

use crate::wgsl::{ self, CoopShape, CoopTile };
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{ Dispatch, backend_extension, tensor::FloatTensor };
use burn::tensor::{ DType, Shape, Tensor as BurnTensor };
use burn_cubecl::{ CubeBackend, kernel::into_contiguous, tensor::CubeTensor };
use cubecl::ir::{ UIntKind, metadata::Info };
use cubecl::prelude::*;
// BufferIOAttr comes with the prelude (cubecl_runtime::kernel::*)
use cubecl::server::KernelArguments;

/// One compiled kernel per source: the id covers the whole text, which is what the compilation caches are keyed by
struct CoopGemmKernel {
    source: String,
    threads: u32,
}

impl KernelMetadata for CoopGemmKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.source.clone())
    }

    fn address_type(&self) -> ElemType {
        UIntKind::U32.into()
    }
}

impl CubeKernel for CoopGemmKernel {
    /// Not compiled (source() is): only the settings of the launch
    fn define(&self) -> KernelDefinition {
        let settings = KernelSettings::new(CubeDim::new_1d(self.threads).0, ExecutionMode::Checked, AddressType::U32);
        KernelDefinition { body: Scope::root(settings.clone()), settings, info: Info::default() }
    }

    fn source(&self) -> Option<PrecompiledSource> {
        Some(PrecompiledSource { source: self.source.clone(), entrypoint_name: "main".to_string(), lang: "wgsl" })
    }
}

fn coop_shape(a: &Shape, b: &Shape) -> Shape {
    Shape::new([a[0].max(b[0]), a[1], b[2]])
}

/// The launch sizes of a [batch, m, k] @ [batch, k, n], an operand of batch 1 shared by every batch
pub fn launch_shape(a: &Shape, b: &Shape) -> CoopShape {
    let (m, k, n) = (a[1], a[2], b[2]);
    CoopShape { m, n, k, stride_a: if a[0] == 1 { 0 } else { m * k }, stride_b: if b[0] == 1 { 0 } else { k * n } }
}

fn launch_coop_gemm(a: CubeTensor, b: CubeTensor, tile: CoopTile) -> CubeTensor {
    let (a, b) = (into_contiguous(a), into_contiguous(b));
    let (sa, sb) = (a.meta.shape().clone(), b.meta.shape().clone());
    assert!(sa[2] == sb[1], "coop gemm: inner dimensions {} and {}", sa[2], sb[1]);
    assert!(sa[0] == sb[0] || sa[0] == 1 || sb[0] == 1, "coop gemm: batches {} and {}", sa[0], sb[0]);
    assert_eq!(a.dtype, DType::F32, "coop gemm: f32 only");
    let shape = coop_shape(&sa, &sb);
    let batch = shape[0];
    let sizes = launch_shape(&sa, &sb);
    let source = wgsl::source(sizes, tile);
    let dtype = a.dtype;
    let buffer = a.client.empty(shape.num_elements() * dtype.size());
    let out = CubeTensor::new_contiguous(a.client.clone(), a.device.clone(), shape, buffer, dtype);
    let count = CubeCount::Static((sizes.n / tile.cols()) as u32, (sizes.m / tile.rows()) as u32, batch as u32);
    // a and b declared read only, so that a tiling that fails to compile marks only its output, not the inputs the
    // test shares with the other tilings
    let args = KernelArguments::new()
        .with_buffer_io(a.handle.clone().binding(), BufferIOAttr::ReadOnly)
        .with_buffer_io(b.handle.clone().binding(), BufferIOAttr::ReadOnly)
        .with_buffer_io(out.handle.clone().binding(), BufferIOAttr::WriteOnly);
    out.client.launch(Box::new(CoopGemmKernel { source, threads: tile.threads() as u32 }), count, args);
    out
}

#[backend_extension(Cube, Fusion)]
pub trait CoopKernels: burn::backend::Backend {
    #[fusion(dtype = a, shape = coop_shape(a, b))]
    fn coop_gemm(a: FloatTensor<Self>, b: FloatTensor<Self>, bk: usize, sgr: usize, sgc: usize, fr: usize, fc: usize) -> FloatTensor<Self>;
}

impl CoopKernels for CubeBackend {
    fn coop_gemm(a: FloatTensor<Self>, b: FloatTensor<Self>, bk: usize, sgr: usize, sgc: usize, fr: usize, fc: usize) -> FloatTensor<Self> {
        launch_coop_gemm(a, b, CoopTile { bk, sgr, sgc, fr, fc })
    }
}

/// [batch, m, k] @ [batch, k, n] -> [batch, m, n] (an operand of batch 1 shared), f32, the sizes multiples of the tile
pub fn gemm(a: BurnTensor<3>, b: BurnTensor<3>, tile: CoopTile) -> BurnTensor<3> {
    let CoopTile { bk, sgr, sgc, fr, fc } = tile;
    BurnTensor::from_dispatch(Dispatch::coop_gemm(a.into_dispatch(), b.into_dispatch(), bk, sgr, sgc, fr, fc))
}
