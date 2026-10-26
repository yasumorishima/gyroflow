// Work branch only (never part of #1143): the cooperative-matrix matmul and convolution (src/coop.rs) on the macOS
// runner's GPU, WGSL on Metal. coop_source: naga validates the WGSL of every code path (vec4 or scalar loads, edges or
// not, matrix or gathered columns) and prints one as MSL. coop_gemm: does the adapter offer cooperative matrices, does
// every tiling match burn's matmul at sizes that are and are not multiples of the tile, and how fast is it at the
// matmul shapes of the model against kernels::gemm. coop_conv: every convolution of the model at the test size (and an
// odd-sized one, batch 2) against burn's conv2d, then timed against what production runs today (kernels::conv2d_implicit,
// burn's conv2d for the 1x1 ones).
//   cargo test --release --test coop -- --ignored --nocapture --exact coop_gemm

use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{ Distribution, Tensor };
use gmflow_coop::coop::wgsl::{ self, CoopB, CoopShape, CoopTile, Win, COOP_DEFAULT };
use gmflow_coop::{ coop, device, diff, kernels };
use std::time::Instant;

const H: usize = 320;
const W: usize = 576;

/// (label, tiling): the default (64 x 64, k 16) and k steps of 8 and 32, and 64 x 128 on 256 threads
const TILES: [(&str, CoopTile); 4] = [
    ("k16 2x2 sg 4x4 frag", COOP_DEFAULT),
    ("k8 2x2 sg 4x4 frag", CoopTile { bk: 8, sgr: 2, sgc: 2, fr: 4, fc: 4 }),
    ("k32 2x2 sg 4x4 frag", CoopTile { bk: 32, sgr: 2, sgc: 2, fr: 4, fc: 4 }),
    ("k16 2x4 sg 4x4 frag", CoopTile { bk: 16, sgr: 2, sgc: 4, fr: 4, fc: 4 }),
];

/// A convolution of the model: (label, c in, c out, kh, kw, stride, ph, pw, map h, map w)
type ConvCase = (&'static str, usize, usize, usize, usize, usize, usize, usize, usize, usize);

fn model_convs() -> Vec<ConvCase> {
    let (h2, w2, h4, w4) = (H / 2, W / 2, H / 4, W / 4);
    vec![
        ("backbone conv1 3->64 7x7 s2", 3, 64, 7, 7, 2, 3, 3, H, W),
        ("layer1 64->64 3x3 @1/2", 64, 64, 3, 3, 1, 1, 1, h2, w2),
        ("layer2 64->96 3x3 s2", 64, 96, 3, 3, 2, 1, 1, h2, w2),
        ("layer2 downsample 64->96 1x1 s2", 64, 96, 1, 1, 2, 0, 0, h2, w2),
        ("layer2 96->96 3x3 @1/4", 96, 96, 3, 3, 1, 1, 1, h4, w4),
        ("layer3 96->128 3x3", 96, 128, 3, 3, 1, 1, 1, h4, w4),
        ("layer3 128->128 3x3", 128, 128, 3, 3, 1, 1, 1, h4, w4),
        ("conv2 128->128 1x1", 128, 128, 1, 1, 1, 0, 0, h4, w4),
        ("trident 128->128 3x3 s2", 128, 128, 3, 3, 2, 1, 1, h4, w4),
        ("convc1 81->256 1x1", 81, 256, 1, 1, 1, 0, 0, h4, w4),
        ("convc2 256->192 3x3", 256, 192, 3, 3, 1, 1, 1, h4, w4),
        ("convf1 2->128 7x7", 2, 128, 7, 7, 1, 3, 3, h4, w4),
        ("convf2 128->64 3x3", 128, 64, 3, 3, 1, 1, 1, h4, w4),
        ("motion conv 256->126 3x3", 256, 126, 3, 3, 1, 1, 1, h4, w4),
        ("GRU 384->128 1x5", 384, 128, 1, 5, 1, 0, 2, h4, w4),
        ("GRU 384->128 5x1", 384, 128, 5, 1, 1, 2, 0, h4, w4),
        ("heads conv1 128->256 3x3", 128, 256, 3, 3, 1, 1, 1, h4, w4),
        ("mask conv2 256->144 1x1", 256, 144, 1, 1, 1, 0, 0, h4, w4),
        ("refine_proj 128->256 1x1", 128, 256, 1, 1, 1, 0, 0, h4, w4),
    ]
}

/// naga (wgpu 30's) parses and validates the WGSL of each path; the first is printed with its MSL. Not a proof that
/// the barriers are in uniform control flow: naga 30 does not check barriers against its uniformity analysis (the edge
/// branch is uniform by construction, its condition from the workgroup id only)
#[test]
#[ignore]
fn coop_source() {
    let conv = |c, h, w, kh, kw, stride, ph, pw| CoopB::Conv { c, h, w, kh, kw, stride, ph, pw };
    let w4 = Win { h: 80, w: 144, splits: 8, shift: true };
    let w8 = Win { h: 40, w: 72, splits: 2, shift: false };
    let shapes = [
        ("conv 256->192 3x3 @1/4, no edges", CoopShape { a_win: None, c_win: None, m: 192, n: 11520, k: 2304, stride_a: 0, b: conv(256, 80, 144, 3, 3, 1, 1, 1) }),
        ("matrix, vec4 a and b, no edges", CoopShape { a_win: None, c_win: None, m: 128, n: 256, k: 96, stride_a: 0, b: CoopB::Matrix { stride: 96 * 256 } }),
        ("matrix, scalar a and b, every edge", CoopShape { a_win: None, c_win: None, m: 100, n: 70, k: 37, stride_a: 100 * 37, b: CoopB::Matrix { stride: 0 } }),
        ("matrix, vec4 a, scalar b, edges in n", CoopShape { a_win: None, c_win: None, m: 64, n: 130, k: 24, stride_a: 0, b: CoopB::Matrix { stride: 24 * 130 } }),
        ("conv 3->64 7x7 s2, edge in k only", CoopShape { a_win: None, c_win: None, m: 64, n: 160 * 288, k: 147, stride_a: 0, b: conv(3, 320, 576, 7, 7, 2, 3, 3) }),
        ("conv 256->126 3x3, edge in m", CoopShape { a_win: None, c_win: None, m: 126, n: 11520, k: 2304, stride_a: 0, b: conv(256, 80, 144, 3, 3, 1, 1, 1) }),
        ("conv 37->45 3x3 on 37x53, every edge", CoopShape { a_win: None, c_win: None, m: 45, n: 37 * 53, k: 333, stride_a: 0, b: conv(37, 37, 53, 3, 3, 1, 1, 1) }),
        ("window scores @1/4, shifted (edges)", CoopShape { m: 180, n: 180, k: 128, stride_a: 0, b: CoopB::WindowT { win: w4 }, a_win: Some(w4), c_win: None }),
        ("window probabilities @ v @1/4, shifted (scattered rows)", CoopShape { m: 180, n: 128, k: 180, stride_a: 180 * 180, b: CoopB::WindowRows { win: w4 }, a_win: None, c_win: Some(w4) }),
        ("window scores @1/8", CoopShape { m: 720, n: 720, k: 128, stride_a: 0, b: CoopB::WindowT { win: w8 }, a_win: Some(w8), c_win: None }),
        ("window probabilities @ v @1/8", CoopShape { m: 720, n: 128, k: 720, stride_a: 720 * 720, b: CoopB::WindowRows { win: w8 }, a_win: None, c_win: Some(w8) }),
    ];
    for (i, (label, shape)) in shapes.iter().enumerate() {
        for (tl, tile) in TILES {
            let src = wgsl::source(*shape, tile);
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}, {tl}: {}\n{src}", e.emit_to_string(&src)));
            let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}, {tl}: {e:?}\n{src}"));
            let options = naga::back::msl::Options { lang_version: (3, 1), ..Default::default() };
            let (msl, _) = naga::back::msl::write_string(&module, &info, &options, &naga::back::msl::PipelineOptions::default())
                .unwrap_or_else(|e| panic!("{label}, {tl}: MSL: {e:?}"));
            println!("SOURCE {label}, {tl}: valid, {} lines of WGSL, {} of MSL", src.lines().count(), msl.lines().count());
            if i == 0 && tile == COOP_DEFAULT {
                println!("{src}\n{msl}");
            }
        }
    }
}

/// The last result is held (one value of it, which keeps its buffer) until the end of the timed calls; 3 calls to warm
/// up (compilation, autotune), then 10 between two device syncs
fn time(device: &burn::tensor::Device, label: &str, flops: f64, f: &dyn Fn() -> Tensor<1>) -> f64 {
    let mut keep = None;
    for _ in 0..3 { keep = Some(f()); }
    device.sync().expect("sync");
    let reps = 10;
    let start = Instant::now();
    for _ in 0..reps { keep = Some(f()); }
    device.sync().expect("sync");
    drop(keep);
    let secs = start.elapsed().as_secs_f64() / reps as f64;
    println!("COOP {label:66} {:9.2} ms {:8.0} GFLOPS", secs * 1e3, flops / secs / 1e9);
    secs
}

/// The device, or None (printed) when its adapter offers no cooperative matrices
fn coop_device() -> Option<burn::tensor::Device> {
    let (device, name, adapter_features, device_features) = device();
    println!("COOP device: {name}");
    let offered = adapter_features.contains("EXPERIMENTAL_COOPERATIVE_MATRIX");
    println!("COOP cooperative matrices offered by the adapter: {offered}, enabled on the device: {}", device_features.contains("EXPERIMENTAL_COOPERATIVE_MATRIX"));
    if !offered {
        println!("COOP adapter features: {adapter_features}");
        println!("COOP nothing to run");
        return None;
    }
    Some(device)
}

#[test]
#[ignore]
fn coop_gemm() {
    let Some(device) = coop_device() else { return };
    let rnd = |s: [usize; 3]| Tensor::<3>::random(s, Distribution::Normal(0.0, 1.0), &device);

    // Against burn's matmul, every tiling. 0.1 % of the largest output: a transposed fragment, a fragment at the wrong
    // place, a wrong batch offset or garbage at an edge is of the size of the outputs themselves
    let check = |label: &str, a: Tensor<3>, b: Tensor<3>| {
        let want = a.clone().matmul(b.clone()).into_data().to_vec::<f32>().expect("f32");
        for (tl, tile) in TILES {
            let got = coop::gemm(a.clone(), b.clone(), tile).into_data().to_vec::<f32>().expect("f32");
            let (max, mean, scale) = diff(&got, &want);
            println!("COOP check {label:40} {tl:22} max |coop - burn| {max:.2e}, mean {mean:.2e} (outputs up to {scale:.2e})");
            assert!(max <= 1e-3 * scale, "{label}, {tl}: max {max:.2e} against outputs up to {scale:.2e}");
        }
    };
    check("[2,128,96] @ [2,96,256]", rnd([2, 128, 96]), rnd([2, 96, 256]));
    check("[1,128,96] @ [2,96,256] (a shared)", rnd([1, 128, 96]), rnd([2, 96, 256]));
    check("[2,128,96] @ [1,96,256] (b shared)", rnd([2, 128, 96]), rnd([1, 96, 256]));
    check("[1,100,37] @ [1,37,70] (every edge)", rnd([1, 100, 37]), rnd([1, 37, 70]));
    check("[2,65,24] @ [1,24,130] (edges, scalar b)", rnd([2, 65, 24]), rnd([1, 24, 130]));
    check("[3,129,31] @ [3,31,67] (edges, batch 3)", rnd([3, 129, 31]), rnd([3, 31, 67]));
    check("[1,100,36] @ [1,36,132] (edges, vec4 b)", rnd([1, 100, 36]), rnd([1, 36, 132]));
    check("[1,192,2304] @ [1,2304,1024]", rnd([1, 192, 2304]), rnd([1, 2304, 1024]));

    let first = |t: Tensor<3>| t.slice([0..1, 0..1, 0..1]).reshape([1]);
    let (l4, l8) = ((H / 4) * (W / 4), (H / 8) * (W / 8));
    // The transformer's linear layers on both images' tokens at once (lin, as burn folds them), and the factored GRU
    let cases = [
        ("FFN fc1 @1/4 [1,2L,256]@[1,256,1024]", rnd([1, 2 * l4, 256]), rnd([1, 256, 1024])),
        ("FFN fc2 @1/4 [1,2L,1024]@[1,1024,128]", rnd([1, 2 * l4, 1024]), rnd([1, 1024, 128])),
        ("q/k/v/merge @1/4 [1,2L,128]@[1,128,128]", rnd([1, 2 * l4, 128]), rnd([1, 128, 128])),
        ("FFN fc1 @1/8 [1,2L,256]@[1,256,1024]", rnd([1, 2 * l8, 256]), rnd([1, 256, 1024])),
        ("GRU factored [1,384,640]@[2,640,L]", rnd([1, 384, 640]), rnd([2, 640, l4])),
        ("square [1,2048,2048]@[1,2048,2048]", rnd([1, 2048, 2048]), rnd([1, 2048, 2048])),
    ];
    for (label, a, b) in cases {
        let (da, db) = (a.dims(), b.dims());
        let flops = 2.0 * (da[0].max(db[0]) * da[1] * da[2] * db[2]) as f64;
        time(&device, &format!("{label}: kernels::gemm"), flops, &|| first(kernels::gemm(a.clone(), b.clone(), None)));
        for (tl, tile) in TILES {
            time(&device, &format!("{label}: {tl}"), flops, &|| first(coop::gemm(a.clone(), b.clone(), tile)));
        }
    }
}

#[test]
#[ignore]
fn coop_conv() {
    let Some(device) = coop_device() else { return };
    let rnd = |s: [usize; 4]| Tensor::<4>::random(s, Distribution::Normal(0.0, 1.0), &device);
    let first = |t: Tensor<4>| t.slice([0..1, 0..1, 0..1, 0..1]).reshape([1]);
    let mut cases = model_convs();
    cases.push(("odd 37->45 3x3 on 37x53, batch 2", 37, 45, 3, 3, 1, 1, 1, 37, 53));
    let mut totals = (0.0, 0.0);
    for (label, cin, cout, kh, kw, stride, ph, pw, h, w) in cases {
        let n = if label.contains("batch 2") { 2 } else { 1 };
        let x = rnd([n, cin, h, w]);
        let wt = rnd([cout, cin, kh, kw]);
        let bias = Tensor::<1>::random([cout], Distribution::Normal(0.0, 1.0), &device);
        let opts = ConvOptions::new([stride, stride], [ph, pw], [1, 1], 1);
        let want = conv2d(x.clone(), wt.clone(), Some(bias.clone()), opts.clone()).into_data().to_vec::<f32>().expect("f32");
        let got = coop::conv2d(x.clone(), wt.clone(), Some(bias.clone()), stride, [ph, pw], COOP_DEFAULT).into_data().to_vec::<f32>().expect("f32");
        let (max, mean, scale) = diff(&got, &want);
        println!("COOP check {label:40} max |coop - burn conv2d| {max:.2e}, mean {mean:.2e} (outputs up to {scale:.2e})");
        assert!(max <= 1e-3 * scale, "{label}: max {max:.2e} against outputs up to {scale:.2e}");
        if n != 1 {
            // The odd one under every tiling too: the gathered columns on 256 threads, the edges of each
            for (tl, tile) in TILES {
                let got = coop::conv2d(x.clone(), wt.clone(), Some(bias.clone()), stride, [ph, pw], tile).into_data().to_vec::<f32>().expect("f32");
                let (max, mean, scale) = diff(&got, &want);
                println!("COOP check {label:40} {tl:22} max |coop - burn conv2d| {max:.2e}, mean {mean:.2e} (outputs up to {scale:.2e})");
                assert!(max <= 1e-3 * scale, "{label}, {tl}: max {max:.2e} against outputs up to {scale:.2e}");
            }
            continue;
        }
        // Against what production runs for it today on the WGSL path: burn for the 1x1 ones, conv2d_implicit (the
        // compile-time sized conv_gemm4_kernel) for the others
        let (oh, ow) = ((h + 2 * ph - kh) / stride + 1, (w + 2 * pw - kw) / stride + 1);
        let flops = 2.0 * (cout * cin * kh * kw * oh * ow) as f64;
        let today = if kh * kw == 1 {
            time(&device, &format!("{label}: burn conv2d (production)"), flops, &|| first(conv2d(x.clone(), wt.clone(), None, opts.clone())))
        } else {
            time(&device, &format!("{label}: conv2d_implicit (production)"), flops, &|| first(kernels::conv2d_implicit(x.clone(), wt.clone(), None, stride, [ph, pw])))
        };
        let coop_secs = time(&device, &format!("{label}: coop"), flops, &|| first(coop::conv2d(x.clone(), wt.clone(), None, stride, [ph, pw], COOP_DEFAULT)));
        totals.0 += today;
        totals.1 += coop_secs;
    }
    println!("COOP the model's convolutions once each: production {:.2} ms, coop {:.2} ms", totals.0 * 1e3, totals.1 * 1e3);
}

/// Split-window attention with its two matmuls on cooperative matrices against kernels::window_attention (itself checked
/// against the plain operations by gmflow-dev's kernels_match_plain), at both sizes of the model, shifted and not, at the
/// one-way and bidirectional batches, then timed
#[test]
#[ignore]
fn coop_attention() {
    let Some(device) = coop_device() else { return };
    let r3 = |s: [usize; 3]| Tensor::<3>::random(s, Distribution::Normal(0.0, 1.0), &device);
    let c = 128;
    for (hh, ww, splits) in [(H / 4, W / 4, 8usize), (H / 8, W / 8, 2)] {
        for bn in [2usize, 4] {
            let (q, k, v) = (r3([bn, hh * ww, c]), r3([bn, hh * ww, c]), r3([bn, hh * ww, c]));
            for shift in [false, true] {
                let want = kernels::window_attention(q.clone(), k.clone(), v.clone(), hh, ww, splits, shift).into_data().to_vec::<f32>().expect("f32");
                let got = coop::window_attention(q.clone(), k.clone(), v.clone(), hh, ww, splits, shift, COOP_DEFAULT).into_data().to_vec::<f32>().expect("f32");
                let (max, mean, scale) = diff(&got, &want);
                println!("COOP check window attention {hh}x{ww}/{splits} b{bn} shift {shift}: max |coop - kernels| {max:.2e}, mean {mean:.2e} (outputs up to {scale:.2e})");
                assert!(max <= 1e-3 * scale.max(1.0), "{hh}x{ww}/{splits} b{bn} shift {shift}: max {max:.2e}");
            }
        }
    }
    let first = |t: Tensor<3>| t.slice([0..1, 0..1, 0..1]).reshape([1]);
    for (hh, ww, splits, bn) in [(H / 4, W / 4, 8usize, 2usize), (H / 4, W / 4, 8, 4), (H / 8, W / 8, 2, 4)] {
        let (q, k, v) = (r3([bn, hh * ww, c]), r3([bn, hh * ww, c]), r3([bn, hh * ww, c]));
        let l = (hh / splits) * (ww / splits);
        let flops = 2.0 * 2.0 * (bn * splits * splits * l * l * c) as f64;
        time(&device, &format!("window attention {hh}x{ww}/{splits} b{bn} shifted: kernels"), flops, &|| first(kernels::window_attention(q.clone(), k.clone(), v.clone(), hh, ww, splits, true)));
        time(&device, &format!("window attention {hh}x{ww}/{splits} b{bn} shifted: coop"), flops, &|| first(coop::window_attention(q.clone(), k.clone(), v.clone(), hh, ww, splits, true, COOP_DEFAULT)));
    }
}
