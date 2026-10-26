// Work branch only (never part of #1143): the WGSL text of the cooperative-matrix matmul and convolution (coop.rs),
// apart from burn so that it can be generated and checked with naga alone (tests/coop.rs prints it, naga validates it
// and writes the MSL).
//
// c[batch] = a[batch] @ b[batch], a [m, k] row-major, b [k, n] either a row-major matrix or the columns of a convolution
// gathered from x [batch, C, H, W] on the fly (F.unfold order: k = (channel, ky, kx), n = (oy, ox)), c [m, n] row-major.
// Each workgroup computes a [rows, cols] block of c: sgr x sgc simdgroups of 32 threads, each fr x fc fragments of 8 x 8;
// a [rows, bk] tile of a and a [bk, cols] tile of b go through workgroup memory, bk columns of a at a time, zero outside
// of the operands. Any m, n, k: a workgroup over an edge of c writes its block to workgroup memory and copies the part
// inside c (whether it does depends on the workgroup only, so the cooperative stores stay in uniform control flow).

use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoopTile {
    pub bk: usize,
    pub sgr: usize,
    pub sgc: usize,
    pub fr: usize,
    pub fc: usize,
}

impl CoopTile {
    pub fn threads(&self) -> usize {
        32 * self.sgr * self.sgc
    }
    /// Rows of c per workgroup
    pub fn rows(&self) -> usize {
        8 * self.fr * self.sgr
    }
    /// Columns of c per workgroup
    pub fn cols(&self) -> usize {
        8 * self.fc * self.sgc
    }
}

/// The tiling the measurements chose (round 7a: 64 x 64, k step 16, 128 threads)
pub const COOP_DEFAULT: CoopTile = CoopTile { bk: 16, sgr: 2, sgc: 2, fr: 4, fc: 4 };

/// Where b comes from
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoopB {
    /// A row-major [k, n] matrix, `stride` elements from one batch to the next (0: shared by every batch)
    Matrix { stride: usize },
    /// The columns of a convolution of x [batch, c, h, w] (kernel kh x kw, stride, zero padding ph, pw): k = c kh kw,
    /// n = oh ow
    Conv { c: usize, h: usize, w: usize, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize },
}

/// The sizes of one launch; `stride_a` 0 for an a shared by every batch
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoopShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub stride_a: usize,
    pub b: CoopB,
}

impl CoopB {
    /// Output size of the convolution, (oh, ow)
    pub fn conv_out(c: usize, h: usize, w: usize, kh: usize, kw: usize, stride: usize, ph: usize, pw: usize) -> (usize, usize) {
        let _ = c;
        ((h + 2 * ph - kh) / stride + 1, (w + 2 * pw - kw) / stride + 1)
    }
}

/// Why `tile` cannot run `shape`, or None
pub fn unsupported(shape: CoopShape, tile: CoopTile) -> Option<String> {
    let t = tile.threads();
    let (wm, wn, bk) = (tile.rows(), tile.cols(), tile.bk);
    if bk == 0 || bk % 8 != 0 {
        return Some(format!("k step {bk}: a positive multiple of 8"));
    }
    // Every thread loads the same number of elements of each tile, and copies the same number at the edges
    if (wm * bk) % t != 0 || (bk * wn) % t != 0 || (wm * wn) % t != 0 {
        return Some(format!("the tiles of {tile:?} do not split evenly over {t} threads"));
    }
    // The gathered columns: each thread keeps one column of the tile (t a multiple of the columns)
    if let CoopB::Conv { c, kh, kw, .. } = shape.b {
        if t % wn != 0 {
            return Some(format!("{tile:?}: {t} threads are not a multiple of its {wn} columns"));
        }
        if c * kh * kw != shape.k {
            return Some(format!("k {} is not c kh kw = {}", shape.k, c * kh * kw));
        }
    }
    // The two tiles (padded rows), an 8 x 8 fragment per simdgroup for the edges and the 64 zeros, in Metal's 32 KiB
    let words = wm * (bk + 4) + bk * (wn + 4) + 64 * tile.sgr * tile.sgc + 64;
    if 4 * words > 32 << 10 {
        return Some(format!("{tile:?} needs {} bytes of workgroup memory, over 32 KiB", 4 * words));
    }
    if shape.m == 0 || shape.n == 0 || shape.k == 0 {
        return Some("an empty matmul".to_string());
    }
    None
}

/// Whether the vec4 loads apply: a row of a, or of b, starts at a multiple of 4 and holds whole vec4s
fn vec_a(shape: CoopShape) -> bool {
    shape.k % 4 == 0 && shape.stride_a % 4 == 0
}
fn vec_b(shape: CoopShape) -> bool {
    match shape.b {
        CoopB::Matrix { stride } => shape.n % 4 == 0 && stride % 4 == 0,
        CoopB::Conv { .. } => false,
    }
}

/// The kernel, entry point `main`, bindings a, b (or x), c in that order
pub fn source(shape: CoopShape, tile: CoopTile) -> String {
    if let Some(why) = unsupported(shape, tile) {
        panic!("coop gemm: {why}");
    }
    let CoopShape { m, n, k, stride_a, b } = shape;
    let CoopTile { bk, sgc, fr, fc, .. } = tile;
    let (t, wm, wn) = (tile.threads(), tile.rows(), tile.cols());
    let (va, vb) = (vec_a(shape), vec_b(shape));
    // Tests that the sizes make always true are left out
    let full_m = m % wm == 0;
    let full_n = n % wn == 0;
    let full_k = k % bk == 0;
    let mut s = String::new();
    let w = &mut s;
    // The writes into a String cannot fail
    macro_rules! emit { ($($arg:tt)*) => { writeln!(w, $($arg)*).unwrap() } }
    emit!("enable wgpu_cooperative_matrix;");
    emit!("");
    // Every binding read_write although a and b are only read: the layout comes from the shader (a precompiled kernel
    // has no representation to take it from), and cubecl places tensors in shared pages, so a, b and c can be slices
    // of one wgpu buffer, which wgpu refuses to bind both read-only and read-write (round 7a; cubecl declares its own
    // kernels' bindings all read-write for that reason, backend/wgsl.rs)
    let ty = |v: bool| if v { "array<vec4<f32>>" } else { "array<f32>" };
    emit!("@group(0) @binding(0) var<storage, read_write> a: {};", ty(va));
    emit!("@group(0) @binding(1) var<storage, read_write> b: {};", ty(vb));
    emit!("@group(0) @binding(2) var<storage, read_write> c: array<f32>;");
    emit!("");
    // The accumulators start from an 8 x 8 block of zeros: loaded, not constructed (what Metal makes of a
    // simdgroup_matrix written as `T {{}}` is not specified)
    emit!("var<workgroup> zeros: array<f32, 64>;");
    let (lda, ldb) = (bk + 4, wn + 4);
    emit!("var<workgroup> tile_a: array<f32, {}>;", wm * lda);
    emit!("var<workgroup> tile_b: array<f32, {}>;", bk * ldb);
    let edge = !(full_m && full_n);
    if edge {
        emit!("var<workgroup> frag: array<f32, {}>;", 64 * tile.sgr * tile.sgc);
    }
    emit!("");
    emit!("@compute @workgroup_size({t}, 1, 1)");
    emit!("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{");
    // Workgroup memory starts uninitialised (cubecl turns wgpu's zeroing off)
    emit!("    for (var z = li; z < 64u; z += {t}u) {{ zeros[z] = 0.0; }}");
    emit!("    workgroupBarrier();");
    // Apple GPUs run 32 threads per simdgroup, numbered in order of the thread index
    emit!("    let sg = li / 32u;");
    emit!("    let sg_row = (sg / {sgc}u) * {}u;", 8 * fr);
    emit!("    let sg_col = (sg % {sgc}u) * {}u;", 8 * fc);
    emit!("    let m0 = wg.y * {wm}u;");
    emit!("    let n0 = wg.x * {wn}u;");
    for i in 0..fr {
        for j in 0..fc {
            emit!("    var c_{i}_{j} = coopLoadT<coop_mat8x8<f32, C>>(&zeros[0], 8u);");
        }
    }
    emit!("    let a_wg = wg.z * {stride_a}u + m0 * {k}u;");
    // The gathered columns: this thread's column of the tile is li % wn at every k step; the top left of its window in
    // x, unsigned (above or left of the map the subtraction wraps around and adding ky or kx wraps back, so one compare
    // per axis tests 0 <= index < size)
    match b {
        CoopB::Matrix { stride } => emit!("    let b_wg = wg.z * {stride}u + n0;"),
        CoopB::Conv { c: cc, h, w: ww, kh, kw, stride, ph, pw } => {
            let (_, ow) = CoopB::conv_out(cc, h, ww, kh, kw, stride, ph, pw);
            emit!("    let x_wg = wg.z * {}u;", cc * h * ww);
            emit!("    let bcol = li % {wn}u;");
            emit!("    let p = n0 + bcol;");
            emit!("    let iy0 = (p / {ow}u) * {stride}u - {ph}u;");
            emit!("    let ix0 = (p % {ow}u) * {stride}u - {pw}u;");
            emit!("    let p_in = p < {n}u;");
        }
    }
    emit!("    for (var k0 = 0u; k0 < {k}u; k0 += {bk}u) {{");
    // The tile of a, [wm, bk]
    if va {
        for it in 0..(wm * bk / 4).div_ceil(t) {
            let guard_t = (it + 1) * t > wm * bk / 4;
            emit!("        {{");
            emit!("            let i = li + {}u;", it * t);
            if guard_t { emit!("            if i < {}u {{", wm * bk / 4); }
            emit!("            let r = i / {}u;", bk / 4);
            emit!("            let q = i % {}u;", bk / 4);
            emit!("            var v = vec4<f32>(0.0);");
            let mut conds = Vec::new();
            if !full_m { conds.push(format!("m0 + r < {m}u")); }
            if !full_k { conds.push(format!("k0 + q * 4u < {k}u")); }
            let load = format!("v = a[(a_wg + r * {k}u + k0) / 4u + q];");
            emit_guarded(w, "            ", &conds, &load);
            emit!("            let o = r * {lda}u + q * 4u;");
            emit!("            tile_a[o] = v.x; tile_a[o + 1u] = v.y; tile_a[o + 2u] = v.z; tile_a[o + 3u] = v.w;");
            if guard_t { emit!("            }}"); }
            emit!("        }}");
        }
    } else {
        for it in 0..wm * bk / t {
            emit!("        {{");
            emit!("            let i = li + {}u;", it * t);
            emit!("            let r = i / {bk}u;");
            emit!("            let q = i % {bk}u;");
            emit!("            var v = 0.0;");
            let mut conds = Vec::new();
            if !full_m { conds.push(format!("m0 + r < {m}u")); }
            if !full_k { conds.push(format!("k0 + q < {k}u")); }
            let load = format!("v = a[a_wg + r * {k}u + k0 + q];");
            emit_guarded(w, "            ", &conds, &load);
            emit!("            tile_a[r * {lda}u + q] = v;");
            emit!("        }}");
        }
    }
    // The tile of b, [bk, wn]
    match b {
        CoopB::Matrix { .. } if vb => {
            for it in 0..(bk * wn / 4).div_ceil(t) {
                let guard_t = (it + 1) * t > bk * wn / 4;
                emit!("        {{");
                emit!("            let i = li + {}u;", it * t);
                if guard_t { emit!("            if i < {}u {{", bk * wn / 4); }
                emit!("            let r = i / {}u;", wn / 4);
                emit!("            let q = i % {}u;", wn / 4);
                emit!("            var v = vec4<f32>(0.0);");
                let mut conds = Vec::new();
                if !full_k { conds.push(format!("k0 + r < {k}u")); }
                if !full_n { conds.push(format!("n0 + q * 4u < {n}u")); }
                let load = format!("v = b[(b_wg + (k0 + r) * {n}u) / 4u + q];");
                emit_guarded(w, "            ", &conds, &load);
                emit!("            let o = r * {ldb}u + q * 4u;");
                emit!("            tile_b[o] = v.x; tile_b[o + 1u] = v.y; tile_b[o + 2u] = v.z; tile_b[o + 3u] = v.w;");
                if guard_t { emit!("            }}"); }
                emit!("        }}");
            }
        }
        CoopB::Matrix { .. } => {
            for it in 0..bk * wn / t {
                emit!("        {{");
                emit!("            let i = li + {}u;", it * t);
                emit!("            let r = i / {wn}u;");
                emit!("            let q = i % {wn}u;");
                emit!("            var v = 0.0;");
                let mut conds = Vec::new();
                if !full_k { conds.push(format!("k0 + r < {k}u")); }
                if !full_n { conds.push(format!("n0 + q < {n}u")); }
                let load = format!("v = b[b_wg + (k0 + r) * {n}u + q];");
                emit_guarded(w, "            ", &conds, &load);
                emit!("            tile_b[r * {ldb}u + q] = v;");
                emit!("        }}");
            }
        }
        CoopB::Conv { h, w: ww, kh, kw, .. } => {
            let khw = kh * kw;
            // Rows of the tile this thread loads: li / wn, then every t / wn rows
            let step = t / wn;
            for it in 0..bk * wn / t {
                emit!("        {{");
                emit!("            let r = li / {wn}u + {}u;", it * step);
                emit!("            let kk = k0 + r;");
                emit!("            var v = 0.0;");
                let mut conds = Vec::new();
                if !full_k { conds.push(format!("kk < {k}u")); }
                if !full_n { conds.push("p_in".to_string()); }
                let mut body = String::new();
                body.push_str(&format!("let ch = kk / {khw}u; let rr = kk % {khw}u; "));
                body.push_str(&format!("let iy = iy0 + rr / {kw}u; let ix = ix0 + rr % {kw}u; "));
                body.push_str(&format!("if iy < {h}u {{ if ix < {ww}u {{ v = b[x_wg + ch * {}u + iy * {ww}u + ix]; }} }}", h * ww));
                emit_guarded(w, "            ", &conds, &body);
                emit!("            tile_b[r * {ldb}u + bcol] = v;");
                emit!("        }}");
            }
        }
    }
    emit!("        workgroupBarrier();");
    for st in 0..bk / 8 {
        for i in 0..fr {
            emit!("        let a_{st}_{i} = coopLoadT<coop_mat8x8<f32, A>>(&tile_a[(sg_row + {}u) * {lda}u + {}u], {lda}u);", i * 8, st * 8);
        }
        for j in 0..fc {
            emit!("        let b_{st}_{j} = coopLoadT<coop_mat8x8<f32, B>>(&tile_b[{}u + sg_col + {}u], {ldb}u);", st * 8 * ldb, j * 8);
        }
        for i in 0..fr {
            for j in 0..fc {
                emit!("        c_{i}_{j} = coopMultiplyAdd(a_{st}_{i}, b_{st}_{j}, c_{i}_{j});");
            }
        }
    }
    emit!("        workgroupBarrier();");
    emit!("    }}");
    emit!("    let c_wg = wg.z * {}u + m0 * {n}u + n0;", m * n);
    let store_direct = |w: &mut String, indent: &str| {
        for i in 0..fr {
            for j in 0..fc {
                writeln!(w, "{indent}coopStoreT(c_{i}_{j}, &c[c_wg + (sg_row + {}u) * {n}u + sg_col + {}u], {n}u);", i * 8, j * 8).unwrap();
            }
        }
    };
    if !edge {
        store_direct(w, "    ");
    } else {
        // Inside c whole or not: the same answer for every thread of the workgroup. Joined by `&`, not `&&`, which naga
        // lowers to a local variable and so to a branch its uniformity analysis takes as non-uniform (naga does not
        // check barriers against that analysis, so validation would not show it)
        let mut inside = Vec::new();
        if !full_m { inside.push(format!("(m0 + {wm}u <= {m}u)")); }
        if !full_n { inside.push(format!("(n0 + {wn}u <= {n}u)")); }
        emit!("    if {} {{", inside.join(" & "));
        store_direct(w, "        ");
        // Over an edge: each fragment into this simdgroup's 8 x 8 of workgroup memory, then its 32 threads copy the 64
        // values (2 each) that are inside c. Every simdgroup stores the same number of fragments, so the barriers are
        // reached by the whole workgroup
        emit!("    }} else {{");
        emit!("        let lane = li % 32u;");
        emit!("        let f0 = sg * 64u;");
        for i in 0..fr {
            for j in 0..fc {
                emit!("        coopStoreT(c_{i}_{j}, &frag[f0], 8u);");
                emit!("        workgroupBarrier();");
                for e in 0..2 {
                    emit!("        {{");
                    emit!("            let e = lane + {}u;", e * 32);
                    emit!("            let r = sg_row + {}u + e / 8u;", i * 8);
                    emit!("            let q = sg_col + {}u + e % 8u;", j * 8);
                    emit!("            if m0 + r < {m}u {{ if n0 + q < {n}u {{ c[c_wg + r * {n}u + q] = frag[f0 + e]; }} }}");
                    emit!("        }}");
                }
                emit!("        workgroupBarrier();");
            }
        }
        emit!("    }}");
    }
    emit!("}}");
    s
}

/// `body` inside nested ifs of `conds` (none: as is)
fn emit_guarded(w: &mut String, indent: &str, conds: &[String], body: &str) {
    if conds.is_empty() {
        writeln!(w, "{indent}{body}").unwrap();
        return;
    }
    let open: String = conds.iter().map(|c| format!("if {c} {{ ")).collect();
    let close: String = conds.iter().map(|_| " }").collect();
    writeln!(w, "{indent}{open}{body}{close}").unwrap();
}
