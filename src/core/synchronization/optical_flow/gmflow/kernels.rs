// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 yasumorishima <fwyasu11@gmail.com>
//
// CubeCL kernels for the parts of gmflow (model.rs) that burn would otherwise run as many small kernels over large
// intermediate tensors: the local correlations (the 9x9 window of matching at 1/4 and of each refinement step), the 3x3
// propagation and the convex upsampling. Each computes in one launch, from the inputs, what model.rs computes with plain
// tensor operations (kept there as the reference the tests compare these to).

use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::{ Dispatch, backend_extension, tensor::FloatTensor };
use burn::tensor::{ Shape, Tensor };
use burn_cubecl::{ CubeBackend, kernel::into_contiguous, tensor::CubeTensor };
use cubecl::{ calculate_cube_count_elemwise, cube, prelude::*, CubeDim };

/// The 9x9 window of matching at 1/4 and of the refinement
pub const CORR_RADIUS: usize = 4;
/// The 3x3 window of the propagation at 1/4
pub const PROP_RADIUS: usize = 1;
pub const UPSAMPLE: usize = 4;

/// local_correlation_with_flow: out[b, k, y, x] = f0[b, :, y, x] . bilinear(f1[b], (x, y) + flow[b, :, y, x] + offset_k)
/// / sqrt(C), zeros outside of the map (each of the 4 taps on its own, as F.grid_sample with zeros padding). One thread
/// per output value.
#[cube(launch)]
fn corr_with_flow_kernel<F: Float>(
    f0: &Tensor<F>,      // [B, C, H, W]
    f1: &Tensor<F>,      // [B, C, H, W]
    flow: &Tensor<F>,    // [B, 2, H, W]
    out: &mut Tensor<F>, // [B, (2r+1)^2, H, W]
    #[comptime] radius: usize,
    #[define(F)] _dtype: ElemType,
) {
    let idx = ABSOLUTE_POS as usize;
    let (b_n, k_n, h, w) = (out.shape(0), out.shape(1), out.shape(2), out.shape(3));
    if idx >= b_n * k_n * h * w {
        terminate!();
    }
    let hw = h * w;
    let x = idx % w;
    let y = (idx / w) % h;
    let k = (idx / hw) % k_n;
    let b = idx / (hw * k_n);
    let n = comptime!(2 * radius + 1);
    let r = comptime!(radius as i32);
    let dx = (k % n) as i32 - r;
    let dy = (k / n) as i32 - r;

    let p = y * w + x;
    let px = F::cast_from(x as i32 + dx) + flow[b * 2 * hw + p];
    let py = F::cast_from(y as i32 + dy) + flow[b * 2 * hw + hw + p];
    let x0f = px.floor();
    let y0f = py.floor();
    let ax = px - x0f;
    let ay = py - y0f;
    let x0 = i32::cast_from(x0f);
    let y0 = i32::cast_from(y0f);
    let wi = w as i32;
    let hi = h as i32;
    let in_x0 = x0 >= 0 && x0 < wi;
    let in_x1 = x0 + 1 >= 0 && x0 + 1 < wi;
    let in_y0 = y0 >= 0 && y0 < hi;
    let in_y1 = y0 + 1 >= 0 && y0 + 1 < hi;
    let one = F::new(1.0);
    let zero = F::new(0.0);
    let w00 = if in_x0 && in_y0 { (one - ax) * (one - ay) } else { zero };
    let w01 = if in_x1 && in_y0 { ax * (one - ay) } else { zero };
    let w10 = if in_x0 && in_y1 { (one - ax) * ay } else { zero };
    let w11 = if in_x1 && in_y1 { ax * ay } else { zero };
    // Clamped addresses: a tap outside of the map has weight 0, its address only has to be valid
    let cx0 = clamp(x0, 0, wi - 1) as usize;
    let cx1 = clamp(x0 + 1, 0, wi - 1) as usize;
    let cy0 = clamp(y0, 0, hi - 1) as usize;
    let cy1 = clamp(y0 + 1, 0, hi - 1) as usize;

    let c_n = f0.shape(1);
    let mut acc = zero;
    for c in 0..c_n {
        let base = (b * c_n + c) * hw;
        let s = w00 * f1[base + cy0 * w + cx0] + w01 * f1[base + cy0 * w + cx1] + w10 * f1[base + cy1 * w + cx0] + w11 * f1[base + cy1 * w + cx1];
        acc += f0[base + p] * s;
    }
    out[idx] = acc / F::sqrt(F::cast_from(c_n as u32));
}

/// local_correlation_softmax at integer offsets: the correlation of each position with the (2r+1)^2 window around it
/// (-1e9 outside of the map), softmax, and the expected sample position minus the position: flow [B, 2, H, W]. One thread
/// per position, online softmax.
#[cube(launch)]
fn local_corr_softmax_kernel<F: Float>(
    f0: &Tensor<F>,      // [B, C, H, W]
    f1: &Tensor<F>,      // [B, C, H, W]
    out: &mut Tensor<F>, // [B, 2, H, W]
    #[comptime] radius: usize,
    #[define(F)] _dtype: ElemType,
) {
    let idx = ABSOLUTE_POS as usize;
    let (b_n, c_n, h, w) = (f0.shape(0), f0.shape(1), f0.shape(2), f0.shape(3));
    if idx >= b_n * h * w {
        terminate!();
    }
    let hw = h * w;
    let p = idx % hw;
    let b = idx / hw;
    let x = p % w;
    let y = p / w;
    let n = comptime!(2 * radius + 1);
    let r = comptime!(radius as i32);
    let scale = F::sqrt(F::cast_from(c_n as u32));
    let mut m = F::new(-1e30);
    let mut s = F::new(0.0);
    let mut ax = F::new(0.0);
    let mut ay = F::new(0.0);
    for k in 0..n * n {
        let sx = x as i32 + (k % n) as i32 - r;
        let sy = y as i32 + (k / n) as i32 - r;
        let valid = sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32;
        let mut score = F::new(-1e9);
        if valid {
            let q = sy as usize * w + sx as usize;
            let mut acc = F::new(0.0);
            for c in 0..c_n {
                let base = (b * c_n + c) * hw;
                acc += f0[base + p] * f1[base + q];
            }
            score = acc / scale;
        }
        let fsx = F::cast_from(sx);
        let fsy = F::cast_from(sy);
        if score > m {
            let carry = F::exp(m - score);
            s = s * carry + F::new(1.0);
            ax = ax * carry + fsx;
            ay = ay * carry + fsy;
            m = score;
        } else {
            let e = F::exp(score - m);
            s += e;
            ax += e * fsx;
            ay += e * fsy;
        }
    }
    out[b * 2 * hw + p] = ax / s - F::cast_from(x as u32);
    out[b * 2 * hw + hw + p] = ay / s - F::cast_from(y as u32);
}

/// SelfAttnPropagation.forward_local_window_attn: softmax over the (2r+1)^2 window of q . k / sqrt(C), the flow of the
/// window weighted by it. Outside of the map the key and the flow are zeros (F.unfold padding): score 0, still counted.
#[cube(launch)]
fn local_propagation_kernel<F: Float>(
    q: &Tensor<F>,       // [B, C, H, W]
    k: &Tensor<F>,       // [B, C, H, W]
    flow: &Tensor<F>,    // [B, 2, H, W]
    out: &mut Tensor<F>, // [B, 2, H, W]
    #[comptime] radius: usize,
    #[define(F)] _dtype: ElemType,
) {
    let idx = ABSOLUTE_POS as usize;
    let (b_n, c_n, h, w) = (q.shape(0), q.shape(1), q.shape(2), q.shape(3));
    if idx >= b_n * h * w {
        terminate!();
    }
    let hw = h * w;
    let p = idx % hw;
    let b = idx / hw;
    let x = p % w;
    let y = p / w;
    let n = comptime!(2 * radius + 1);
    let r = comptime!(radius as i32);
    let scale = F::sqrt(F::cast_from(c_n as u32));
    let mut m = F::new(-1e30);
    let mut s = F::new(0.0);
    let mut ax = F::new(0.0);
    let mut ay = F::new(0.0);
    for j in 0..n * n {
        let sx = x as i32 + (j % n) as i32 - r;
        let sy = y as i32 + (j / n) as i32 - r;
        let valid = sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32;
        let mut score = F::new(0.0);
        let mut vx = F::new(0.0);
        let mut vy = F::new(0.0);
        if valid {
            let o = sy as usize * w + sx as usize;
            let mut acc = F::new(0.0);
            for c in 0..c_n {
                let base = (b * c_n + c) * hw;
                acc += q[base + p] * k[base + o];
            }
            score = acc / scale;
            vx = flow[b * 2 * hw + o];
            vy = flow[b * 2 * hw + hw + o];
        }
        if score > m {
            let carry = F::exp(m - score);
            s = s * carry + F::new(1.0);
            ax = ax * carry + vx;
            ay = ay * carry + vy;
            m = score;
        } else {
            let e = F::exp(score - m);
            s += e;
            ax += e * vx;
            ay += e * vy;
        }
    }
    out[b * 2 * hw + p] = ax / s;
    out[b * 2 * hw + hw + p] = ay / s;
}

/// upsample_flow_with_mask: out[b, c, f*h+ky, f*w+kx] = sum over the 3x3 window j of softmax_j(mask[b, j*f*f + ky*f+kx, h, w])
/// * f * flow[b, c, (h, w) + offset_j] (zeros outside). One thread per output position, both channels.
#[cube(launch)]
fn convex_upsample_kernel<F: Float>(
    flow: &Tensor<F>,    // [B, 2, H, W]
    mask: &Tensor<F>,    // [B, 9 * f * f, H, W]
    out: &mut Tensor<F>, // [B, 2, f*H, f*W]
    #[comptime] factor: usize,
    #[define(F)] _dtype: ElemType,
) {
    let idx = ABSOLUTE_POS as usize;
    let (b_n, h, w) = (flow.shape(0), flow.shape(2), flow.shape(3));
    let ow = w * factor;
    let oh = h * factor;
    if idx >= b_n * oh * ow {
        terminate!();
    }
    let ohw = oh * ow;
    let op = idx % ohw;
    let b = idx / ohw;
    let ox = op % ow;
    let oy = op / ow;
    let x = ox / factor;
    let y = oy / factor;
    let sub = (oy % factor) * factor + ox % factor;
    let hw = h * w;
    let p = y * w + x;
    let ff = comptime!(factor * factor);
    let mbase = b * 9 * ff * hw + sub * hw + p;
    let mut m = F::new(-1e30);
    for j in 0..9usize {
        m = F::max(m, mask[mbase + j * ff * hw]);
    }
    let mult = F::cast_from(factor as u32);
    let mut s = F::new(0.0);
    let mut ax = F::new(0.0);
    let mut ay = F::new(0.0);
    for j in 0..9usize {
        let e = F::exp(mask[mbase + j * ff * hw] - m);
        s += e;
        let sx = x as i32 + (j % 3) as i32 - 1;
        let sy = y as i32 + (j / 3) as i32 - 1;
        if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
            let o = sy as usize * w + sx as usize;
            ax += e * (mult * flow[b * 2 * hw + o]);
            ay += e * (mult * flow[b * 2 * hw + hw + o]);
        }
    }
    out[b * 2 * ohw + op] = ax / s;
    out[b * 2 * ohw + ohw + op] = ay / s;
}

// ---------------------------------------------------------------- launches

fn corr_shape(f0: &Shape) -> Shape {
    let n = 2 * CORR_RADIUS + 1;
    Shape::new([f0[0], n * n, f0[2], f0[3]])
}

fn flow_shape(f0: &Shape) -> Shape {
    Shape::new([f0[0], 2, f0[2], f0[3]])
}

fn upsampled_shape(flow: &Shape) -> Shape {
    Shape::new([flow[0], 2, flow[2] * UPSAMPLE, flow[3] * UPSAMPLE])
}

/// An uninitialised contiguous output on the device of `like`, and the launch grid for one thread per `threads`
fn output_for(like: &CubeTensor, shape: Shape, threads: usize) -> (CubeTensor, cubecl::CubeCount, CubeDim) {
    let dtype = like.dtype;
    let buffer = like.client.empty(shape.num_elements() * dtype.size());
    let out = CubeTensor::new_contiguous(like.client.clone(), like.device.clone(), shape, buffer, dtype);
    let cube_dim = CubeDim::new(&like.client, threads);
    let cube_count = calculate_cube_count_elemwise(&like.client, threads, cube_dim);
    (out, cube_count, cube_dim)
}

fn launch_corr_with_flow(f0: CubeTensor, f1: CubeTensor, flow: CubeTensor) -> CubeTensor {
    let (f0, f1, flow) = (into_contiguous(f0), into_contiguous(f1), into_contiguous(flow));
    let shape = corr_shape(f0.meta.shape());
    let n = shape.num_elements();
    let dtype = f0.dtype;
    let (out, count, dim) = output_for(&f0, shape, n);
    corr_with_flow_kernel::launch(&out.client, count, dim, f0.into_tensor_arg(), f1.into_tensor_arg(), flow.into_tensor_arg(), out.clone().into_tensor_arg(), CORR_RADIUS, dtype_to_storage_type(dtype));
    out
}

fn launch_local_corr_softmax(f0: CubeTensor, f1: CubeTensor) -> CubeTensor {
    let (f0, f1) = (into_contiguous(f0), into_contiguous(f1));
    let s = f0.meta.shape().clone();
    let n = s[0] * s[2] * s[3];
    let dtype = f0.dtype;
    let (out, count, dim) = output_for(&f0, flow_shape(&s), n);
    local_corr_softmax_kernel::launch(&out.client, count, dim, f0.into_tensor_arg(), f1.into_tensor_arg(), out.clone().into_tensor_arg(), CORR_RADIUS, dtype_to_storage_type(dtype));
    out
}

fn launch_local_propagation(q: CubeTensor, k: CubeTensor, flow: CubeTensor) -> CubeTensor {
    let (q, k, flow) = (into_contiguous(q), into_contiguous(k), into_contiguous(flow));
    let s = flow.meta.shape().clone();
    let n = s[0] * s[2] * s[3];
    let dtype = flow.dtype;
    let (out, count, dim) = output_for(&flow, s, n);
    local_propagation_kernel::launch(&out.client, count, dim, q.into_tensor_arg(), k.into_tensor_arg(), flow.into_tensor_arg(), out.clone().into_tensor_arg(), PROP_RADIUS, dtype_to_storage_type(dtype));
    out
}

fn launch_convex_upsample(flow: CubeTensor, mask: CubeTensor) -> CubeTensor {
    let (flow, mask) = (into_contiguous(flow), into_contiguous(mask));
    let shape = upsampled_shape(flow.meta.shape());
    let n = shape[0] * shape[2] * shape[3];
    let dtype = flow.dtype;
    let (out, count, dim) = output_for(&flow, shape, n);
    convex_upsample_kernel::launch(&out.client, count, dim, flow.into_tensor_arg(), mask.into_tensor_arg(), out.clone().into_tensor_arg(), UPSAMPLE, dtype_to_storage_type(dtype));
    out
}

// ---------------------------------------------------------------- the backend extension

/// The kernels as backend operations: on the CubeCL backends (Vulkan, Metal), and lazily under burn fusion
#[backend_extension(Cube, Fusion)]
pub trait GmflowKernels: burn::backend::Backend {
    #[fusion(dtype = f0, shape = corr_shape(f0))]
    fn corr_with_flow(f0: FloatTensor<Self>, f1: FloatTensor<Self>, flow: FloatTensor<Self>) -> FloatTensor<Self>;
    #[fusion(dtype = f0, shape = flow_shape(f0))]
    fn local_corr_softmax(f0: FloatTensor<Self>, f1: FloatTensor<Self>) -> FloatTensor<Self>;
    #[fusion(dtype = flow, shape = flow)]
    fn local_propagation(q: FloatTensor<Self>, k: FloatTensor<Self>, flow: FloatTensor<Self>) -> FloatTensor<Self>;
    #[fusion(dtype = flow, shape = upsampled_shape(flow))]
    fn convex_upsample(flow: FloatTensor<Self>, mask: FloatTensor<Self>) -> FloatTensor<Self>;
}

impl GmflowKernels for CubeBackend {
    fn corr_with_flow(f0: FloatTensor<Self>, f1: FloatTensor<Self>, flow: FloatTensor<Self>) -> FloatTensor<Self> {
        launch_corr_with_flow(f0, f1, flow)
    }
    fn local_corr_softmax(f0: FloatTensor<Self>, f1: FloatTensor<Self>) -> FloatTensor<Self> {
        launch_local_corr_softmax(f0, f1)
    }
    fn local_propagation(q: FloatTensor<Self>, k: FloatTensor<Self>, flow: FloatTensor<Self>) -> FloatTensor<Self> {
        launch_local_propagation(q, k, flow)
    }
    fn convex_upsample(flow: FloatTensor<Self>, mask: FloatTensor<Self>) -> FloatTensor<Self> {
        launch_convex_upsample(flow, mask)
    }
}

// ---------------------------------------------------------------- what model.rs calls

/// f0, f1 [B, C, H, W], flow [B, 2, H, W] -> [B, 81, H, W]
pub fn corr_with_flow(f0: Tensor<4>, f1: Tensor<4>, flow: Tensor<4>) -> Tensor<4> {
    Tensor::from_dispatch(Dispatch::corr_with_flow(f0.into_dispatch(), f1.into_dispatch(), flow.into_dispatch()))
}

/// f0, f1 [B, C, H, W] -> flow [B, 2, H, W]
pub fn local_corr_softmax(f0: Tensor<4>, f1: Tensor<4>) -> Tensor<4> {
    Tensor::from_dispatch(Dispatch::local_corr_softmax(f0.into_dispatch(), f1.into_dispatch()))
}

/// q, k [B, C, H, W] (projected), flow [B, 2, H, W] -> flow
pub fn local_propagation(q: Tensor<4>, k: Tensor<4>, flow: Tensor<4>) -> Tensor<4> {
    Tensor::from_dispatch(Dispatch::local_propagation(q.into_dispatch(), k.into_dispatch(), flow.into_dispatch()))
}

/// flow [B, 2, H, W], mask [B, 144, H, W] -> [B, 2, 4H, 4W]
pub fn convex_upsample(flow: Tensor<4>, mask: Tensor<4>) -> Tensor<4> {
    Tensor::from_dispatch(Dispatch::convex_upsample(flow.into_dispatch(), mask.into_dispatch()))
}
