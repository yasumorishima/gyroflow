// Work branch only (never part of #1143): the WGSL text of the cooperative-matrix matmul (coop.rs), apart from burn so
// that it can be generated and checked with naga alone (tests/coop.rs prints it, naga validates it and writes the MSL).

use std::fmt::Write;

/// How the work is split: `sgr` x `sgc` simdgroups of 32 threads per workgroup, each computing `fr` x `fc` fragments of
/// 8 x 8 outputs; `bk` 0 loads the fragments of a and b straight from the buffers, otherwise a [rows, bk] tile of a and
/// a [bk, columns] tile of b go through workgroup memory first, `bk` columns of a at a time
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
    /// Rows of the output per workgroup
    pub fn rows(&self) -> usize {
        8 * self.fr * self.sgr
    }
    /// Columns of the output per workgroup
    pub fn cols(&self) -> usize {
        8 * self.fc * self.sgc
    }
}

/// The sizes of one launch: c[batch] = a[batch] @ b[batch], a [m, k], b [k, n], c [m, n], row-major and contiguous; a
/// batch stride of 0 for an operand of batch 1 (shared by every batch)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoopShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub stride_a: usize,
    pub stride_b: usize,
}

/// Why `tile` cannot compute `shape` (no edge handling: the sizes must be multiples of the tile), or None
pub fn unsupported(shape: CoopShape, tile: CoopTile) -> Option<String> {
    let CoopShape { m, n, k, .. } = shape;
    let t = tile.threads();
    if m % tile.rows() != 0 || n % tile.cols() != 0 || k % 8 != 0 {
        return Some(format!("[{m}, {k}] @ [{k}, {n}] is not a multiple of the tile {tile:?}"));
    }
    if tile.bk > 0 {
        if tile.bk % 8 != 0 || k % tile.bk != 0 || n % 4 != 0 {
            return Some(format!("k {k} or n {n} does not fit the k step {}", tile.bk));
        }
        // Each thread loads whole vec4s, the same number of them
        if (tile.rows() * tile.bk / 4) % t != 0 || (tile.bk * tile.cols() / 4) % t != 0 {
            return Some(format!("the tiles of {tile:?} do not split evenly over {t} threads"));
        }
        if 4 * (tile.rows() * (tile.bk + 4) + tile.bk * (tile.cols() + 4)) > 32 << 10 {
            return Some(format!("{tile:?} needs more than 32 KiB of workgroup memory"));
        }
    }
    None
}

/// The kernel, entry point `main`, bindings a, b, c in that order (a and b read, c written)
pub fn source(shape: CoopShape, tile: CoopTile) -> String {
    if let Some(why) = unsupported(shape, tile) {
        panic!("coop gemm: {why}");
    }
    let CoopShape { m, n, k, stride_a, stride_b } = shape;
    let CoopTile { bk, sgc, fr, fc, .. } = tile;
    let (t, wm, wn) = (tile.threads(), tile.rows(), tile.cols());
    let mut s = String::new();
    let w = &mut s;
    // The writes into a String cannot fail
    macro_rules! emit { ($($arg:tt)*) => { writeln!(w, $($arg)*).unwrap() } }
    emit!("enable wgpu_cooperative_matrix;");
    emit!("");
    if bk == 0 {
        emit!("@group(0) @binding(0) var<storage, read> a: array<f32>;");
        emit!("@group(0) @binding(1) var<storage, read> b: array<f32>;");
    } else {
        emit!("@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;");
        emit!("@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;");
    }
    emit!("@group(0) @binding(2) var<storage, read_write> c: array<f32>;");
    emit!("");
    // The accumulators start from an 8 x 8 block of zeros: loaded, not constructed (what Metal makes of a
    // simdgroup_matrix written as `T {{}}` is not specified)
    emit!("var<workgroup> zeros: array<f32, 64>;");
    let (lda, ldb) = (bk + 4, wn + 4);
    if bk > 0 {
        emit!("var<workgroup> tile_a: array<f32, {}>;", wm * lda);
        emit!("var<workgroup> tile_b: array<f32, {}>;", bk * ldb);
    }
    emit!("");
    emit!("@compute @workgroup_size({t}, 1, 1)");
    emit!("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{");
    emit!("    if li < 64u {{ zeros[li] = 0.0; }}");
    emit!("    workgroupBarrier();");
    // Apple GPUs run 32 threads per simdgroup, numbered in order of the thread index
    emit!("    let sg = li / 32u;");
    emit!("    let sg_row = (sg / {sgc}u) * {}u;", 8 * fr);
    emit!("    let sg_col = (sg % {sgc}u) * {}u;", 8 * fc);
    emit!("    let row0 = wg.y * {wm}u + sg_row;");
    emit!("    let col0 = wg.x * {wn}u + sg_col;");
    for i in 0..fr {
        for j in 0..fc {
            emit!("    var c_{i}_{j} = coopLoadT<coop_mat8x8<f32, C>>(&zeros[0], 8u);");
        }
    }
    if bk == 0 {
        emit!("    let a0 = wg.z * {stride_a}u + row0 * {k}u;");
        emit!("    let b0 = wg.z * {stride_b}u + col0;");
        emit!("    for (var kk = 0u; kk < {k}u; kk += 8u) {{");
        for i in 0..fr {
            emit!("        let a_{i} = coopLoadT<coop_mat8x8<f32, A>>(&a[a0 + {}u + kk], {k}u);", i * 8 * k);
        }
        for j in 0..fc {
            emit!("        let b_{j} = coopLoadT<coop_mat8x8<f32, B>>(&b[b0 + kk * {n}u + {}u], {n}u);", j * 8);
        }
        for i in 0..fr {
            for j in 0..fc {
                emit!("        c_{i}_{j} = coopMultiplyAdd(a_{i}, b_{j}, c_{i}_{j});");
            }
        }
        emit!("    }}");
    } else {
        // Element offsets of the first row of a and of the first column of b of the workgroup (multiples of 4: k and n
        // are, and so are the batch strides)
        emit!("    let a_wg = wg.z * {stride_a}u + wg.y * {}u;", wm * k);
        emit!("    let b_wg = wg.z * {stride_b}u + wg.x * {wn}u;");
        emit!("    for (var k0 = 0u; k0 < {k}u; k0 += {bk}u) {{");
        let (a_vec, b_vec) = (wm * bk / 4, bk * wn / 4);
        for it in 0..a_vec / t {
            emit!("        {{");
            emit!("            let i = li + {}u;", it * t);
            emit!("            let r = i / {}u;", bk / 4);
            emit!("            let q = i % {}u;", bk / 4);
            emit!("            let v = a[(a_wg + r * {k}u + k0) / 4u + q];");
            emit!("            let o = r * {lda}u + q * 4u;");
            emit!("            tile_a[o] = v.x; tile_a[o + 1u] = v.y; tile_a[o + 2u] = v.z; tile_a[o + 3u] = v.w;");
            emit!("        }}");
        }
        for it in 0..b_vec / t {
            emit!("        {{");
            emit!("            let i = li + {}u;", it * t);
            emit!("            let r = i / {}u;", wn / 4);
            emit!("            let q = i % {}u;", wn / 4);
            emit!("            let v = b[(b_wg + (k0 + r) * {n}u) / 4u + q];");
            emit!("            let o = r * {ldb}u + q * 4u;");
            emit!("            tile_b[o] = v.x; tile_b[o + 1u] = v.y; tile_b[o + 2u] = v.z; tile_b[o + 3u] = v.w;");
            emit!("        }}");
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
    }
    emit!("    let c0 = wg.z * {}u + row0 * {n}u + col0;", m * n);
    for i in 0..fr {
        for j in 0..fc {
            emit!("    coopStoreT(c_{i}_{j}, &c[c0 + {}u], {n}u);", i * 8 * n + j * 8);
        }
    }
    emit!("}}");
    s
}
