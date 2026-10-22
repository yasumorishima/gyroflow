// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 yasumorishima <fwyasu11@gmail.com>
//
// CubeCL kernels for the parts of gmflow (model.rs) that burn would otherwise run as many small kernels over large
// intermediate tensors: the local correlations (the 9x9 window of matching at 1/4 and of each refinement step), the 3x3
// propagation and the convex upsampling. Each computes in one launch, from the inputs, what model.rs computes with plain
// tensor operations (kept there as the reference the tests compare these to).

use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{ Dispatch, backend_extension, tensor::{ FloatTensor, IntTensor } };
// burn::tensor::Int is the kind of integer tensors; Int in kernels is the CubeCL trait
// Tensor in the kernels is the CubeCL one: the burn tensors of the functions model.rs calls are BurnTensor
use burn::tensor::{ DType, Int as IntKind, Shape, Tensor as BurnTensor };
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

// ---------------------------------------------------------------- convolution as im2col + matmul

/// The columns of a convolution (F.unfold): out[n, (c * KH + ky) * KW + kx, oy * OW + ox] = x[n, c, oy * s - ph + ky,
/// ox * s - pw + kx], zero outside. Multiplied by the weights [Cout, C*KH*KW], they give the convolution. One thread per
/// output value.
#[cube(launch)]
fn im2col_kernel<F: Float>(
    x: &Tensor<F>,       // [N, C, H, W]
    out: &mut Tensor<F>, // [N, C*KH*KW, OH*OW]
    #[comptime] kh: usize,
    #[comptime] kw: usize,
    #[comptime] stride: usize,
    #[comptime] ph: usize,
    #[comptime] pw: usize,
    #[define(F)] _dtype: ElemType,
) {
    let idx = ABSOLUTE_POS as usize;
    let (n_n, k_n, l_n) = (out.shape(0), out.shape(1), out.shape(2));
    if idx >= n_n * k_n * l_n {
        terminate!();
    }
    let (c_n, h, w) = (x.shape(1), x.shape(2), x.shape(3));
    let ow = (w + 2 * pw - kw) / stride + 1;
    let l = idx % l_n;
    let k = (idx / l_n) % k_n;
    let n = idx / (l_n * k_n);
    let kx = k % kw;
    let ky = (k / kw) % kh;
    let c = k / (kw * kh);
    let iy = ((l / ow) * stride + ky) as i32 - ph as i32;
    let ix = ((l % ow) * stride + kx) as i32 - pw as i32;
    let mut v = F::new(0.0);
    if iy >= 0 && iy < h as i32 && ix >= 0 && ix < w as i32 {
        v = x[((n * c_n + c) * h + iy as usize) * w + ix as usize];
    }
    out[idx] = v;
}

fn im2col_shape(x: &Shape, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize) -> Shape {
    let oh = (x[2] + 2 * ph - kh) / stride + 1;
    let ow = (x[3] + 2 * pw - kw) / stride + 1;
    Shape::new([x[0], x[1] * kh * kw, oh * ow])
}

fn launch_im2col(x: CubeTensor, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize) -> CubeTensor {
    let x = into_contiguous(x);
    let shape = im2col_shape(x.meta.shape(), kh, kw, stride, ph, pw);
    let n = shape.num_elements();
    let dtype = x.dtype;
    let (out, count, dim) = output_for(&x, shape, n);
    im2col_kernel::launch(&out.client, count, dim, x.into_tensor_arg(), out.clone().into_tensor_arg(), kh, kw, stride, ph, pw, dtype_to_storage_type(dtype));
    out
}

// ---------------------------------------------------------------- preprocessing on the GPU

/// A frame of 8-bit grey (luma), packed 3 bytes to an i32 word (always positive, so that no conversion can change it), as
/// it is uploaded: [H, words per row]
pub fn pack_frame(width: usize, height: usize, stride: usize, pixels: &[u8]) -> (Vec<i32>, [usize; 2]) {
    let words = width.div_ceil(3);
    let mut out = vec![0i32; height * words];
    for y in 0..height {
        let row = &pixels[y * stride..y * stride + width];
        for (j, chunk) in row.chunks(3).enumerate() {
            let mut v = 0i32;
            for (b, &p) in chunk.iter().enumerate() { v |= (p as i32) << (8 * b); }
            out[y * words + j] = v;
        }
    }
    (out, [height, words])
}

/// What `resize` of the image crate (0.25) does with FilterType::Triangle, then the letterbox: the frame scaled to
/// new_w x new_h, vertical pass then horizontal pass with the weights of the triangle filter widened by the ratio and
/// normalised, rounded to an integer grey level once at the end, placed at (pad_x, pad_y) on a canvas of zeros; the same
/// grey in the 3 channels: [1, 3, out_h, out_w], raw 0-255 as the model takes it. One thread per canvas position.
#[cube(launch)]
fn preprocess_kernel<I: Int, F: Float>(
    frame: &Tensor<I>,   // [H0, ceil(W0 / 3)], 3 bytes per word
    out: &mut Tensor<F>, // [1, 3, out_h, out_w]
    #[comptime] w0: usize,
    #[comptime] new_w: usize,
    #[comptime] new_h: usize,
    #[comptime] pad_x: usize,
    #[comptime] pad_y: usize,
    #[define(I)] _itype: ElemType,
    #[define(F)] _ftype: ElemType,
) {
    let idx = ABSOLUTE_POS as usize;
    let (oh, ow) = (out.shape(2), out.shape(3));
    if idx >= oh * ow {
        terminate!();
    }
    let ox = idx % ow;
    let oy = idx / ow;
    let h0 = frame.shape(0);
    let words = frame.shape(1);
    let zero = F::new(0.0);
    let one = F::new(1.0);
    let half = F::new(0.5);
    let mut value = zero;
    if ox >= pad_x && ox < pad_x + new_w && oy >= pad_y && oy < pad_y + new_h {
        let x = ox - pad_x;
        let y = oy - pad_y;
        // Vertical window and weights (vertical_sample)
        let ry = F::cast_from(h0 as u32) / F::cast_from(new_h as u32);
        let sy = F::max(ry, one);
        let iny = (F::cast_from(y as u32) + half) * ry;
        let top = clamp(i32::cast_from((iny - sy).floor()), 0, h0 as i32 - 1);
        let bottom = clamp(i32::cast_from((iny + sy).ceil()), top + 1, h0 as i32);
        let iny = iny - half;
        let mut sumy = zero;
        for i in top..bottom {
            let d = ((F::cast_from(i) - iny) / sy).abs();
            if d < one { sumy += one - d; }
        }
        // Horizontal window and weights (horizontal_sample)
        let rx = F::cast_from(w0 as u32) / F::cast_from(new_w as u32);
        let sx = F::max(rx, one);
        let inx = (F::cast_from(x as u32) + half) * rx;
        let left = clamp(i32::cast_from((inx - sx).floor()), 0, w0 as i32 - 1);
        let right = clamp(i32::cast_from((inx + sx).ceil()), left + 1, w0 as i32);
        let inx = inx - half;
        let mut sumx = zero;
        for j in left..right {
            let d = ((F::cast_from(j) - inx) / sx).abs();
            if d < one { sumx += one - d; }
        }
        let mut t = zero;
        for j in left..right {
            let dx = ((F::cast_from(j) - inx) / sx).abs();
            let mut wx = zero;
            if dx < one { wx = one - dx; }
            wx = wx / sumx;
            // The vertical pass at column j
            let mut v = zero;
            for i in top..bottom {
                let dy = ((F::cast_from(i) - iny) / sy).abs();
                let mut wy = zero;
                if dy < one { wy = one - dy; }
                wy = wy / sumy;
                let word = u32::cast_from(frame[i as usize * words + j as usize / 3]);
                let byte = (word >> (8 * (j as u32 % 3))) & 255;
                v += F::cast_from(byte) * wy;
            }
            t += v * wx;
        }
        // f32::round of the image crate rounds half away from zero; WGSL round() rounds half to even: t >= 0, so
        // floor(t + 0.5) is what the CPU does
        value = (clamp(t, zero, F::new(255.0)) + half).floor();
    }
    let plane = oh * ow;
    let v = value;
    out[idx] = v;
    out[plane + idx] = v;
    out[2 * plane + idx] = v;
}

fn preprocess_meta(_frame: &TensorSpec, _w0: &usize, _new_w: &usize, _new_h: &usize, _pad_x: &usize, _pad_y: &usize, out_h: &usize, out_w: &usize) -> TensorSpec {
    TensorSpec::new(Shape::new([1, 3, *out_h, *out_w]), DType::F32)
}

#[allow(clippy::too_many_arguments)]
fn launch_preprocess(frame: CubeTensor, w0: usize, new_w: usize, new_h: usize, pad_x: usize, pad_y: usize, out_h: usize, out_w: usize) -> CubeTensor {
    let frame = into_contiguous(frame);
    let shape = Shape::new([1, 3, out_h, out_w]);
    let threads = out_h * out_w;
    let dtype = DType::F32;
    let buffer = frame.client.empty(shape.num_elements() * dtype.size());
    let out = CubeTensor::new_contiguous(frame.client.clone(), frame.device.clone(), shape, buffer, dtype);
    let cube_dim = CubeDim::new(&frame.client, threads);
    let count = calculate_cube_count_elemwise(&frame.client, threads, cube_dim);
    let itype = dtype_to_storage_type(frame.dtype);
    preprocess_kernel::launch(&out.client, count, cube_dim, frame.into_tensor_arg(), out.clone().into_tensor_arg(), w0, new_w, new_h, pad_x, pad_y, itype, dtype_to_storage_type(dtype));
    out
}

/// The letterbox of a frame packed by `pack_frame` (width w0), computed on the GPU: [1, 3, out_h, out_w] raw 0-255
#[allow(clippy::too_many_arguments)]
pub fn preprocess(frame: BurnTensor<2, IntKind>, w0: usize, new_w: usize, new_h: usize, pad_x: usize, pad_y: usize, out_h: usize, out_w: usize) -> BurnTensor<4> {
    BurnTensor::from_dispatch(Dispatch::preprocess(frame.into_dispatch(), w0, new_w, new_h, pad_x, pad_y, out_h, out_w))
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
    #[fusion(dtype = x, shape = im2col_shape(x, *kh, *kw, *stride, *ph, *pw))]
    fn im2col(x: FloatTensor<Self>, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize) -> FloatTensor<Self>;
    #[fusion(meta = preprocess_meta)]
    #[allow(clippy::too_many_arguments)]
    fn preprocess(frame: IntTensor<Self>, w0: usize, new_w: usize, new_h: usize, pad_x: usize, pad_y: usize, out_h: usize, out_w: usize) -> FloatTensor<Self>;
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
    fn im2col(x: FloatTensor<Self>, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize) -> FloatTensor<Self> {
        launch_im2col(x, kh, kw, stride, ph, pw)
    }
    fn preprocess(frame: IntTensor<Self>, w0: usize, new_w: usize, new_h: usize, pad_x: usize, pad_y: usize, out_h: usize, out_w: usize) -> FloatTensor<Self> {
        launch_preprocess(frame, w0, new_w, new_h, pad_x, pad_y, out_h, out_w)
    }
}

// ---------------------------------------------------------------- what model.rs calls

/// f0, f1 [B, C, H, W], flow [B, 2, H, W] -> [B, 81, H, W]
pub fn corr_with_flow(f0: BurnTensor<4>, f1: BurnTensor<4>, flow: BurnTensor<4>) -> BurnTensor<4> {
    BurnTensor::from_dispatch(Dispatch::corr_with_flow(f0.into_dispatch(), f1.into_dispatch(), flow.into_dispatch()))
}

/// f0, f1 [B, C, H, W] -> flow [B, 2, H, W]
pub fn local_corr_softmax(f0: BurnTensor<4>, f1: BurnTensor<4>) -> BurnTensor<4> {
    BurnTensor::from_dispatch(Dispatch::local_corr_softmax(f0.into_dispatch(), f1.into_dispatch()))
}

/// q, k [B, C, H, W] (projected), flow [B, 2, H, W] -> flow
pub fn local_propagation(q: BurnTensor<4>, k: BurnTensor<4>, flow: BurnTensor<4>) -> BurnTensor<4> {
    BurnTensor::from_dispatch(Dispatch::local_propagation(q.into_dispatch(), k.into_dispatch(), flow.into_dispatch()))
}

/// flow [B, 2, H, W], mask [B, 144, H, W] -> [B, 2, 4H, 4W]
pub fn convex_upsample(flow: BurnTensor<4>, mask: BurnTensor<4>) -> BurnTensor<4> {
    BurnTensor::from_dispatch(Dispatch::convex_upsample(flow.into_dispatch(), mask.into_dispatch()))
}

/// The columns of x [N, C, H, W] for a KH x KW convolution (F.unfold): [N, C*KH*KW, OH*OW]
pub fn im2col(x: BurnTensor<4>, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::im2col(x.into_dispatch(), kh, kw, stride, ph, pw))
}

/// A 2D convolution (symmetric zero padding, no dilation or groups) as im2col + one batched matmul, the columns made in
/// chunks of images so that none is over `max_bytes` (Vulkan guarantees 128 MiB for one buffer). x [N, C, H, W],
/// weight [Cout, C, KH, KW] -> [N, Cout, OH, OW]
pub fn conv2d_im2col(x: BurnTensor<4>, weight: BurnTensor<4>, bias: Option<BurnTensor<1>>, stride: usize, padding: [usize; 2], max_bytes: usize) -> BurnTensor<4> {
    let [n, c, h, w] = x.dims();
    let [cout, cin, kh, kw] = weight.dims();
    assert_eq!(c, cin, "conv2d_im2col: channels");
    let (ph, pw) = (padding[0], padding[1]);
    let oh = (h + 2 * ph - kh) / stride + 1;
    let ow = (w + 2 * pw - kw) / stride + 1;
    let k = c * kh * kw;
    let per_image = k * oh * ow * 4;
    assert!(per_image <= max_bytes, "conv2d_im2col: the columns of one image ({per_image} bytes) exceed {max_bytes}");
    let chunk = (max_bytes / per_image).max(1);
    let wmat = weight.reshape([1, cout, k]);
    let mut outs = Vec::new();
    let mut start = 0;
    while start < n {
        let len = chunk.min(n - start);
        let part = if len == n { x.clone() } else { x.clone().narrow(0, start, len) };
        let cols: BurnTensor<3> = BurnTensor::from_dispatch(Dispatch::im2col(part.into_dispatch(), kh, kw, stride, ph, pw));
        outs.push(wmat.clone().matmul(cols)); // [len, Cout, OH*OW]
        start += len;
    }
    let out = if outs.len() == 1 { outs.pop().unwrap() } else { BurnTensor::cat(outs, 0) };
    let out = out.reshape([n, cout, oh, ow]);
    match bias {
        Some(b) => out + b.reshape([1, cout, 1, 1]),
        None => out,
    }
}
