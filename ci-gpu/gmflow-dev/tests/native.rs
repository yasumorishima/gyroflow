// The hand-written gmflow against PyTorch (refs from ci-gpu/refs_native.py). Run one at a time, ignored by default:
//   GMFLOW_PTH=gmflow.pth GMFLOW_BPK=out.bpk cargo test --release --test native -- --ignored --exact convert_weights
//   GMFLOW_BPK=out.bpk GMFLOW_REF_DIR=refs cargo test --release --test native -- --ignored --exact matches_pytorch

use burn::module::Module;
use burn_store::{ BurnpackStore, ModuleSnapshot, PytorchStore };
use gmflow_dev::model::{ self, Gmflow, Statics, Trace };
use gmflow_dev::kernels;
use burn::tensor::{ Distribution, Tensor };
use gmflow_dev::*;
use std::path::PathBuf;
use std::time::Instant;

const H: usize = 320;
const W: usize = 576;

/// The tilings of kernels::gemm the ceiling test times (block / k step / outputs per thread per side), the default first;
/// kernels_match_plain checks every one of them against burn's matmul
const GEMM_TILES: [kernels::GemmTile; 5] = [
    kernels::GEMM_DEFAULT,
    kernels::GemmTile { block: 64, kstep: 16, per: 4 },
    kernels::GemmTile { block: 32, kstep: 16, per: 2 },
    kernels::GemmTile { block: 32, kstep: 32, per: 2 },
    kernels::GemmTile { block: 128, kstep: 8, per: 8 },
];

/// The configurations measured: (label, ops)
fn configs() -> Vec<(&'static str, model::Ops)> {
    let o = |kernels, conv_im2col, gru_factored, own_gemm, gemm_linear| model::Ops { kernels, conv_im2col, gru_factored, own_gemm, gemm_linear };
    vec![
        ("plain", o(false, false, false, false, false)),
        ("kernels", o(true, false, false, false, false)),
        ("kernels+gru", o(true, false, true, false, false)),
        ("kernels+im2col", o(true, true, false, false, false)),
        ("kernels+im2col+gru", o(true, true, true, false, false)),
        ("kernels+im2col+gru+gemm", o(true, true, true, true, false)),
        ("kernels+im2col+gru+gemm+linear", o(true, true, true, true, true)),
    ]
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} not set")))
}

#[test]
#[ignore]
fn convert_weights() {
    let (device, name) = device();
    println!("device: {name}");
    let mut model = Gmflow::new(&device);
    let mut store = PytorchStore::from_file(env_path("GMFLOW_PTH"))
        .with_top_level_key("model")
        .map_indices_contiguous(false)
        .with_key_remapping(r"\.mlp\.0\.", ".mlp.fc1.")
        .with_key_remapping(r"\.mlp\.2\.", ".mlp.fc2.")
        .with_key_remapping(r"^refine\.mask\.0\.", "refine.mask.conv1.")
        .with_key_remapping(r"^refine\.mask\.2\.", "refine.mask.conv2.")
        .with_key_remapping(r"\.downsample\.0\.", ".downsample.");
    let applied = model.load_from(&mut store).expect("load the checkpoint");
    println!("applied {}, skipped {}, missing {:?}, unused {:?}, errors {:?}", applied.applied.len(), applied.skipped.len(), applied.missing, applied.unused, applied.errors);
    assert!(applied.missing.is_empty() && applied.unused.is_empty() && applied.errors.is_empty(), "the checkpoint does not match the model");
    println!("parameters: {}", model.num_params());
    let out = env_path("GMFLOW_BPK");
    let mut bpk = BurnpackStore::from_file(&out).overwrite(true);
    model.save_into(&mut bpk).expect("write the weights");
    println!("wrote {:?}", out);
}

fn load(device: &burn::tensor::Device) -> Gmflow {
    let mut model = Gmflow::new(device);
    let mut store = BurnpackStore::from_file(env_path("GMFLOW_BPK"));
    let applied = model.load_from(&mut store).expect("load the weights");
    assert!(applied.missing.is_empty() && applied.unused.is_empty() && applied.errors.is_empty(), "weights do not match: {:?} {:?} {:?}", applied.missing, applied.unused, applied.errors);
    model
}

#[test]
#[ignore]
fn matches_pytorch() {
    let (device, name) = device();
    println!("device: {name}");
    let dir = env_path("GMFLOW_REF_DIR");
    let model = load(&device);
    let mut st = Statics::new(H, W, &device);
    // The stages with the plain tensor operations (the port itself); the kernels are checked against these separately
    st.ops.kernels = false;

    // Stage by stage on pair 1, against the intermediates of unimatch (reported, not gated: errors carry over)
    let inp = |k: usize, i: usize| tensor4(&dir.join(format!("pair{k}_in{i}.f32")), [1, 3, H, W], &device);
    let mut trace = Trace::default();
    let _ = model.forward_traced(inp(1, 0), inp(1, 1), &st, Some(&mut trace));
    let manifest = std::fs::read_to_string(dir.join("inter").join("manifest.txt")).expect("manifest");
    let shapes: std::collections::HashMap<String, Vec<usize>> = manifest.lines().map(|l| {
        let (n, s) = l.split_once(' ').unwrap();
        (n.to_owned(), s.split(',').map(|d| d.parse().unwrap()).collect())
    }).collect();
    for (stage, t) in trace.tensors {
        let Some(shape) = shapes.get(&stage) else { println!("stage {stage:22}: no reference"); continue };
        assert_eq!(&t.dims().to_vec(), shape, "stage {stage}: shape");
        let reference = read_f32(&dir.join("inter").join(format!("{stage}.f32")));
        let (max, mean, scale) = diff(&to_vec(t), &reference);
        println!("stage {stage:22}: max |burn - torch| {max:.2e}, mean {mean:.2e} (values up to {scale:.2e})");
    }

    // The gate (pre-registered for the ONNX model, the same here): every pair, max <= 1e-2 px and mean <= 1e-4 px, with the
    // plain operations and with the kernels
    for (label, ops) in configs() {
        st.ops = ops;
        let mut checked = 0;
        for k in 0.. {
            if !dir.join(format!("pair{k}_torch.f32")).exists() { break; }
            let started = Instant::now();
            let flow = to_vec(model.forward(inp(k, 0), inp(k, 1), &st));
            let secs = started.elapsed().as_secs_f64();
            let (max, mean, _) = diff(&flow, &read_f32(&dir.join(format!("pair{k}_torch.f32"))));
            println!("{label:30} pair {k}: max |burn - torch| {max:.2e} px, mean {mean:.2e} px, one pass {secs:.3} s");
            assert!(max <= 1e-2 && mean <= 1e-4, "pair {k}: max {max:e}, mean {mean:e}");
            checked += 1;
        }
        assert!(checked >= 3, "found {checked} reference pairs");
    }

    // Bidirectional, each frame through the CNN on its own (as the tracker will): the flow both ways against unimatch with
    // pred_bidir_flow, the same gate
    // With the kernels as configured, and without them (which part breaks, if one does); all printed before the assert
    let im2col = std::env::var("GMFLOW_IM2COL").map(|v| v == "1").unwrap_or(true);
    let mut bad = Vec::new();
    for (what, kernels_on) in [("kernels", true), ("plain ops", false)] {
        st.ops = model::Ops { kernels: kernels_on, conv_im2col: im2col, gru_factored: true, own_gemm: im2col && kernels_on, gemm_linear: im2col && kernels_on };
        for k in 0..3 {
            let fa = model.features(inp(k, 0), &st);
            let fb = model.features(inp(k, 1), &st);
            let started = Instant::now();
            let both = model.flow(&fa, &fb, true, &st, None);
            let fwd = to_vec(both.clone().narrow(0, 0, 1));
            let bwd = to_vec(both.narrow(0, 1, 1));
            let secs = started.elapsed().as_secs_f64();
            for (dir, got, file) in [("a -> b", fwd, format!("pair{k}_torch.f32")), ("b -> a", bwd, format!("pair{k}_bwd_torch.f32"))] {
                let (max, mean, _) = diff(&got, &read_f32(&env_path("GMFLOW_REF_DIR").join(&file)));
                println!("bidirectional ({what}) pair {k} {dir}: max |burn - torch| {max:.2e} px, mean {mean:.2e} px ({secs:.3} s for both, without the CNN)");
                if !(max <= 1e-2 && mean <= 1e-4) { bad.push(format!("{what} pair {k} {dir}: max {max:e}, mean {mean:e}")); }
            }
        }
    }
    st.ops = model::Ops { kernels: true, conv_im2col: im2col, gru_factored: true, own_gemm: im2col, gemm_linear: im2col };
    assert!(bad.is_empty(), "bidirectional: {bad:?}");

    // The tracker, per frame: before, two one-way passes (each through the CNN with both frames); now, the CNN of the new
    // frame and one bidirectional pass. Warm, mean of 5 after 2
    {
        let (i0, i1) = (inp(1, 0), inp(1, 1));
        let time = |f: &dyn Fn()| { for _ in 0..2 { f(); } let t = Instant::now(); for _ in 0..5 { f(); } t.elapsed().as_secs_f64() / 5.0 };
        let before = time(&|| { let _ = to_vec(model.forward(i0.clone(), i1.clone(), &st)); let _ = to_vec(model.forward(i1.clone(), i0.clone(), &st)); });
        let fa = model.features(i0.clone(), &st);
        let after = time(&|| { let fb = model.features(i1.clone(), &st); let _ = to_vec(model.flow(&fa, &fb, true, &st, None)); });
        println!("tracker per frame ({:?}): two one-way passes {before:.3} s, CNN of one frame + bidirectional pass {after:.3} s", st.ops);
    }

    // Warm: the same pair again until the time settles (spread of the last 5 under 5 %) or 30 passes
    let runs: usize = std::env::var("GMFLOW_WARM_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    if runs == 0 { return; }
    for (label, ops) in configs() {
    st.ops = ops;
    let mut times = Vec::new();
    for _ in 0..30 {
        let started = Instant::now();
        let _ = to_vec(model.forward(inp(1, 0), inp(1, 1), &st));
        times.push(started.elapsed().as_secs_f64());
        if times.len() >= runs.max(5) {
            let last = &times[times.len() - 5..];
            let mean = last.iter().sum::<f64>() / 5.0;
            let sd = (last.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / 5.0).sqrt();
            if sd / mean < 0.05 { break; }
        }
    }
    let last = &times[times.len() - 5..];
    let mean = last.iter().sum::<f64>() / 5.0;
    println!("warm, {label}: {mean:.3} s per pass (mean of the last 5 of {} passes: {:?})", times.len(), last.iter().map(|t| format!("{t:.3}")).collect::<Vec<_>>());
    }
}

/// Each kernel against the plain tensor operations it replaces, on random inputs of the real sizes. Limit set before the
/// first run: max |kernel - plain| <= 1e-4 (values of order 1; a wrong index or tap gives errors of order 1)
#[test]
#[ignore]
fn kernels_match_plain() {
    let (device, name) = device();
    println!("device: {name}");
    // Batch 2, as the bidirectional pass runs the kernels
    let (b, c, h, w) = (2usize, 128usize, H / 4, W / 4);
    let rnd = |shape: [usize; 4], sd: f64| Tensor::<4>::random(shape, Distribution::Normal(0.0, sd), &device);
    let f0 = rnd([b, c, h, w], 1.0);
    let f1 = rnd([b, c, h, w], 1.0);
    // Flows of a few pixels, so that the taps cross the borders of the map
    let flow = rnd([b, 2, h, w], 3.0);
    // Each check against its limit. local_correlation_softmax returns sample coordinates (up to 143 at 1/4) minus the
    // position, as unimatch computes the flow: float32 rounding of numbers that size alone is about 1e-4 (measured 1.07e-4
    // at batch 2, over the 1e-4 set for every check before measuring), so its limit is 1e-3, still far under the 1 px a
    // wrong offset would give
    let mut failed = Vec::new();
    let mut check = |label: &str, got: Tensor<4>, want: Tensor<4>| {
        assert_eq!(got.dims(), want.dims(), "{label}: shape");
        let (max, mean, scale) = diff(&to_vec(got), &to_vec(want));
        // local_correlation_with_flow: the plain op rounds the sample position x + dx + flow (up to ~160, float32 step
        // 1.5e-5) before taking its fraction, the kernel takes the fraction of the flow alone; with neighbouring values of
        // random features differing by up to ~10 that is ~1e-4 (measured 1.10e-4 for flows just under integers on
        // lavapipe, after the 1e-4 set before measuring): 1e-3 as for local_correlation_softmax, still far under the
        // errors of order 1 of a wrong tap
        let rounding_of_positions = label == "local_correlation_softmax" || label.contains("corr_with_flow") || label == "local_correlation_with_flow";
        let limit = if rounding_of_positions { 1e-3 } else { 1e-4 };
        println!("{label:28}: max |kernel - plain| {max:.2e}, mean {mean:.2e} (values up to {scale:.2e}, limit {limit:e})");
        if !(max <= limit) { failed.push(format!("{label}: {max:e}")); }
    };
    // As a run of the model calls it: batch 1 first (one-way passes), then batch 2 (bidirectional), and flows of tens of
    // pixels as at 1/4 on real pairs
    for (bn, sd) in [(1usize, 3.0f64), (2, 3.0), (2, 20.0), (1, 20.0)] {
        let (g0, g1) = (rnd([bn, c, h, w], 1.0), rnd([bn, c, h, w], 1.0));
        let fl = rnd([bn, 2, h, w], sd);
        let label = format!("corr_with_flow b{bn} sd{sd}");
        check(&label, kernels::corr_with_flow(g0.clone(), g1.clone(), fl.clone()), model::local_correlation_with_flow(g0, g1, fl, kernels::CORR_RADIUS));
    }
    check("local_correlation_with_flow", kernels::corr_with_flow(f0.clone(), f1.clone(), flow.clone()), model::local_correlation_with_flow(f0.clone(), f1.clone(), flow.clone(), kernels::CORR_RADIUS));
    // The kernel takes the bilinear weights from the fractional part of the flow alone: integer flows (weight 0 on the
    // second taps) and flows just under an integer (weight ~1 on them) as well
    let flow_int = flow.clone().round();
    check("corr_with_flow, integer flow", kernels::corr_with_flow(f0.clone(), f1.clone(), flow_int.clone()), model::local_correlation_with_flow(f0.clone(), f1.clone(), flow_int.clone(), kernels::CORR_RADIUS));
    let flow_under = flow_int.sub_scalar(1e-3);
    check("corr_with_flow, just under", kernels::corr_with_flow(f0.clone(), f1.clone(), flow_under.clone()), model::local_correlation_with_flow(f0.clone(), f1.clone(), flow_under, kernels::CORR_RADIUS));
    check("local_correlation_softmax", kernels::local_corr_softmax(f0.clone(), f1.clone()), model::local_correlation_softmax(f0.clone(), f1.clone(), kernels::CORR_RADIUS));
    check("local propagation", kernels::local_propagation(f0.clone(), f1.clone(), flow.clone()), model::local_propagation(f0.clone(), f1.clone(), flow.clone(), kernels::PROP_RADIUS));
    let mask = rnd([b, 9 * 16, h, w], 2.0);
    check("convex upsampling", kernels::convex_upsample(flow.clone(), mask.clone()), model::convex_upsample(flow, mask));
    // Convolutions of the shapes of the model as im2col + matmul, against burn
    for (cin, cout, kh, kw, stride, ph, pw, hh, ww) in [(3usize, 64usize, 7usize, 7usize, 2usize, 3usize, 3usize, H, W), (64, 64, 3, 3, 1, 1, 1, H / 2, W / 2), (256, 192, 3, 3, 1, 1, 1, h, w), (384, 128, 1, 5, 1, 0, 2, h, w), (384, 128, 5, 1, 1, 2, 0, h, w)] {
        let x = rnd([2, cin, hh, ww], 1.0);
        let wt = rnd([cout, cin, kh, kw], 0.02); // outputs of order 1, as the other checks
        let bias = Tensor::<1>::random([cout], Distribution::Normal(0.0, 0.1), &device);
        let want = burn::tensor::module::conv2d(x.clone(), wt.clone(), Some(bias.clone()), burn::tensor::ops::ConvOptions::new([stride, stride], [ph, pw], [1, 1], 1));
        let got = kernels::conv2d_im2col(x.clone(), wt.clone(), Some(bias.clone()), stride, [ph, pw], 120 << 20, false);
        check(&format!("conv {cin}->{cout} {kh}x{kw}/{stride} im2col"), got, want.clone());
        let got = kernels::conv2d_im2col(x, wt, Some(bias), stride, [ph, pw], 120 << 20, true);
        check(&format!("conv {cin}->{cout} {kh}x{kw}/{stride} im2col gemm"), got, want);
    }
    // kernels::gemm against burn's matmul, with every tiling the ceiling test times (the default first): the shapes of the
    // model (a of 1 shared by the batch) and both batched, some with rows and columns not multiples of the 32, 64 or 128
    // block and k not a multiple of the 8, 16 or 32 step (every tiling meets partial tiles in all three); a scaled by 1/sqrt(K) so that the outputs are of order 1
    let rnd3 = |shape: [usize; 3], sd: f64| Tensor::<3>::random(shape, Distribution::Normal(0.0, sd), &device);
    assert_eq!(GEMM_TILES[0], kernels::GEMM_DEFAULT, "the default tiling comes first");
    for (ba, bb, m, k, nn) in [(1usize, 1usize, 192usize, 2304usize, 11520usize), (1, 2, 126, 1152, 11520), (1, 2, 384, 640, 11520), (1, 1, 64, 147, 46080), (2, 2, 180, 128, 180), (1, 2, 2, 1152, 11520), (2, 1, 100, 256, 500)] {
        let a = rnd3([ba, m, k], 1.0 / (k as f64).sqrt());
        let bm = rnd3([bb, k, nn], 1.0);
        let want = a.clone().matmul(bm.clone()).unsqueeze_dim::<4>(0);
        check(&format!("gemm [{ba},{m},{k}]x[{bb},{k},{nn}]"), kernels::gemm(a.clone(), bm.clone(), None).unsqueeze_dim::<4>(0), want.clone());
        // gemm is gemm_tiled with GEMM_TILES[0]: the others
        for tile in &GEMM_TILES[1..] {
            let tile = *tile;
            let label = format!("gemm {}/{}/{} [{ba},{m},{k}]x[{bb},{k},{nn}]", tile.block, tile.kstep, tile.per);
            check(&label, kernels::gemm_tiled(a.clone(), bm.clone(), None, tile).unsqueeze_dim::<4>(0), want.clone());
        }
    }
    {
        let a = rnd3([1, 100, 300], 0.06);
        let bm = rnd3([2, 300, 333], 1.0);
        let bias = Tensor::<1>::random([100], Distribution::Normal(0.0, 0.1), &device);
        let want = (a.clone().matmul(bm.clone()) + bias.clone().reshape([1, 100, 1])).unsqueeze_dim::<4>(0);
        check("gemm with bias [1,100,300]x[2,300,333]", kernels::gemm(a.clone(), bm.clone(), Some(bias.clone())).unsqueeze_dim::<4>(0), want.clone());
        for tile in &GEMM_TILES[1..] {
            let tile = *tile;
            let label = format!("gemm {}/{}/{} with bias [1,100,300]x[2,300,333]", tile.block, tile.kstep, tile.per);
            check(&label, kernels::gemm_tiled(a.clone(), bm.clone(), Some(bias.clone()), tile).unsqueeze_dim::<4>(0), want.clone());
        }
    }
    assert!(failed.is_empty(), "kernels that differ from the plain operations: {failed:?}");
}

/// The letterbox the CPU made before (image crate resize, Triangle), as gmflow.rs computes it
fn cpu_preprocess(img: &image::GrayImage) -> (Vec<f32>, [usize; 5]) {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let scale = (W as f32 / iw).min(H as f32 / ih);
    let new_w = ((iw * scale).round() as usize).min(W).max(1);
    let new_h = ((ih * scale).round() as usize).min(H).max(1);
    let (pad_x, pad_y) = ((W - new_w) / 2, (H - new_h) / 2);
    let resized = image::imageops::resize(img, new_w as u32, new_h as u32, image::imageops::FilterType::Triangle);
    let mut data = vec![0f32; 3 * H * W];
    for y in 0..new_h {
        for x in 0..new_w {
            let v = resized.as_raw()[y * new_w + x] as f32;
            for c in 0..3 { data[c * H * W + (y + pad_y) * W + x + pad_x] = v; }
        }
    }
    (data, [img.width() as usize, new_w, new_h, pad_x, pad_y])
}

/// The GPU letterbox against the CPU one, on textured frames of several sizes and row strides. Limits set before the first
/// run (plan): max |gpu - cpu| <= 1 grey level, mean <= 0.05 (the CPU rounds to an integer level once; a rounding that
/// falls on the other side of .5 by float32 differences is one level)
#[test]
#[ignore]
fn preprocess_matches_cpu() {
    let (device, name) = device();
    println!("device: {name}");
    for (w0, h0, stride) in [(1280usize, 720usize, 1280usize), (960, 540, 960), (640, 480, 640), (576, 320, 576), (1001, 563, 1024), (320, 180, 320)] {
        let mut pixels = vec![0u8; stride * h0];
        for y in 0..h0 {
            for x in 0..w0 {
                let v = 128.0 + 60.0 * ((x as f32) * 0.13).sin() * ((y as f32) * 0.07).cos() + 50.0 * (((x * 7 + y * 13) % 31) as f32 / 31.0 - 0.5);
                pixels[y * stride + x] = v.round().clamp(0.0, 255.0) as u8;
            }
            for x in w0..stride { pixels[y * stride + x] = 255; } // what lies past the width must not be read
        }
        let mut tight = Vec::with_capacity(w0 * h0);
        for y in 0..h0 { tight.extend_from_slice(&pixels[y * stride..y * stride + w0]); }
        let img = image::GrayImage::from_raw(w0 as u32, h0 as u32, tight).unwrap();
        let (want, [_, new_w, new_h, pad_x, pad_y]) = cpu_preprocess(&img);
        let (words, shape) = kernels::pack_frame(w0, h0, stride, &pixels);
        let frame = Tensor::<2, burn::tensor::Int>::from_data(burn::tensor::TensorData::new(words, shape), &device);
        let got = to_vec(kernels::preprocess(frame, w0, new_w, new_h, pad_x, pad_y, H, W));
        let (max, mean, _) = diff(&got, &want);
        let off = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("{w0}x{h0} (stride {stride}) -> {new_w}x{new_h} at ({pad_x}, {pad_y}): max |gpu - cpu| {max}, mean {mean:.2e}, {off} of {} values differ", got.len());
        assert!(max <= 1.0 && mean <= 0.05, "{w0}x{h0}: max {max}, mean {mean}");
    }
}

/// Three warm passes of one configuration (GMFLOW_PROFILE_OPS, a label of `configs`) between markers, for cubecl's
/// per-kernel profile (CUBECL_DEBUG_OPTION=profile-medium): what each kernel costs. Times of a profiled run are not speeds
#[test]
#[ignore]
fn profile_passes() {
    let (device, name) = device();
    println!("device: {name}");
    let dir = env_path("GMFLOW_REF_DIR");
    let model = load(&device);
    let mut st = Statics::new(H, W, &device);
    let want = std::env::var("GMFLOW_PROFILE_OPS").expect("GMFLOW_PROFILE_OPS");
    st.ops = configs().into_iter().find(|(l, _)| *l == want).unwrap_or_else(|| panic!("no configuration {want}")).1;
    let inp = |i: usize| tensor4(&dir.join(format!("pair1_in{i}.f32")), [1, 3, H, W], &device);
    for _ in 0..2 { let _ = to_vec(model.forward(inp(0), inp(1), &st)); }
    std::thread::sleep(std::time::Duration::from_secs(3));
    println!("PROFILE-START {want}");
    for _ in 0..3 { let _ = to_vec(model.forward(inp(0), inp(1), &st)); }
    std::thread::sleep(std::time::Duration::from_secs(3));
    println!("PROFILE-END");
}

/// Where the time of a pass goes, stage by stage (the GPU is waited for at each trace point, so the stages do not overlap
/// and fusion stops at their borders): one pass on a pair, and the bidirectional pass of the tracker on CNN features
#[test]
#[ignore]
fn stage_times() {
    let (device, name) = device();
    println!("device: {name}");
    let dir = env_path("GMFLOW_REF_DIR");
    let model = load(&device);
    let mut st = Statics::new(H, W, &device);
    let want = std::env::var("GMFLOW_PROFILE_OPS").expect("GMFLOW_PROFILE_OPS");
    st.ops = configs().into_iter().find(|(l, _)| *l == want).unwrap_or_else(|| panic!("no configuration {want}")).1;
    let inp = |i: usize| tensor4(&dir.join(format!("pair1_in{i}.f32")), [1, 3, H, W], &device);
    // Stage label: the trace name, without the iteration number for the refinement, so that its 6 steps add up
    let label = |n: &str| {
        let base = n.trim_end_matches(|c: char| c.is_ascii_digit());
        if ["corr_with_flow", "refine_net", "refine_delta"].contains(&base) { base.to_owned() } else { n.to_owned() }
    };
    let report = |what: &str, runs: Vec<(Instant, Vec<(String, Instant)>)>| {
        let mut order: Vec<String> = Vec::new();
        let mut sums: std::collections::HashMap<String, f64> = Default::default();
        let mut total = 0.0;
        for (start, marks) in &runs {
            let mut prev = *start;
            for (n, t) in marks {
                let l = label(n);
                if !order.contains(&l) { order.push(l.clone()); }
                *sums.entry(l).or_default() += (*t - prev).as_secs_f64();
                prev = *t;
            }
            total += (prev - *start).as_secs_f64();
        }
        let k = runs.len() as f64;
        println!("{what} ({want}): {:.3} s per pass with a wait at each stage (mean of {})", total / k, runs.len());
        for l in order {
            let v = sums[&l] / k;
            println!("  {l:22} {:8.1} ms  {:5.1} %", v * 1e3, 100.0 * v * k / total);
        }
    };
    let timed = || Trace { timed: true, ..Default::default() };
    let mut runs = Vec::new();
    for i in 0..5 {
        let (a, b) = (inp(0), inp(1));
        let _ = to_vec(a.clone() + b.clone());
        device.sync().expect("sync");
        let mut tr = timed();
        let start = Instant::now();
        let _ = to_vec(model.forward_traced(a, b, &st, Some(&mut tr)));
        tr.marks.push(("read of the flow".to_owned(), Instant::now()));
        if i >= 2 { runs.push((start, tr.marks)); }
    }
    report("one pass", runs);
    let fa = model.features(inp(0), &st);
    let fb = model.features(inp(1), &st);
    device.sync().expect("sync");
    let mut runs = Vec::new();
    for i in 0..5 {
        let mut tr = timed();
        let start = Instant::now();
        let _ = to_vec(model.flow(&fa, &fb, true, &st, Some(&mut tr)));
        tr.marks.push(("read of the flow".to_owned(), Instant::now()));
        if i >= 2 { runs.push((start, tr.marks)); }
    }
    report("bidirectional flow, CNN features given", runs);
}

/// How fast burn's matmuls are on this GPU at the shapes of the model, against the arithmetic peak (independent
/// multiply-add chains): whether the passes are bound by the matmul kernels or by the FLOPs themselves. Each case: 3 calls
/// to warm up (autotune runs on the first), then 10 calls timed between two device syncs
#[test]
#[ignore]
fn matmul_ceiling() {
    let (device, name) = device();
    println!("device: {name}");
    // What gmflow.rs decides the convolutions from (im2col_is_faster)
    if let burn::backend::DispatchDevice::Cube(d) = device.as_dispatch() {
        let client = d.client();
        let m = &client.properties().features.matmul;
        for c in m.cmma.iter() { println!("CEIL cmma {:?} x {:?} -> {:?} ({}x{}x{})", c.a_type, c.b_type, c.cd_type, c.m, c.n, c.k); }
        for c in m.mma.iter() { println!("CEIL mma {:?} x {:?} -> {:?} ({}x{}x{})", c.a_type, c.b_type, c.cd_type, c.m, c.n, c.k); }
        println!("CEIL matrix units: {} cmma and {} mma configurations", m.cmma.len(), m.mma.len());
    }
    let rnd2 = |s: [usize; 2]| Tensor::<2>::random(s, Distribution::Normal(0.0, 1.0), &device);
    let rnd3 = |s: [usize; 3]| Tensor::<3>::random(s, Distribution::Normal(0.0, 1.0), &device);
    let rnd4 = |s: [usize; 4]| Tensor::<4>::random(s, Distribution::Normal(0.0, 1.0), &device);
    // The last result is held (one value of it, which keeps its buffer) until the end of the timed calls
    let time = |label: &str, flops: f64, f: &dyn Fn() -> Tensor<1>| {
        let mut keep = None;
        for _ in 0..3 { keep = Some(f()); }
        device.sync().expect("sync");
        let reps = 10;
        let start = Instant::now();
        for _ in 0..reps { keep = Some(f()); }
        device.sync().expect("sync");
        drop(keep);
        let secs = start.elapsed().as_secs_f64() / reps as f64;
        println!("CEIL {label:58} {:9.2} ms {:8.0} GFLOPS", secs * 1e3, flops / secs / 1e9);
    };
    let first = |t: Tensor<3>| t.slice([0..1, 0..1, 0..1]).reshape([1]);
    let first2 = |t: Tensor<2>| t.slice([0..1, 0..1]).reshape([1]);
    let first4 = |t: Tensor<4>| t.slice([0..1, 0..1, 0..1, 0..1]).reshape([1]);
    // One image: the columns of two (212 MB) would be over the 128 MiB of one buffer
    let (n, cout, k, l) = (1usize, 192usize, 256 * 9, (H / 4) * (W / 4));
    let conv_flops = 2.0 * (n * cout * k * l) as f64;
    let w = rnd3([1, cout, k]);
    let cols = rnd3([n, k, l]);
    time("conv 256->192 3x3 @1/4: [1,Co,K]@[1,K,L] (as the model)", conv_flops, &|| first(w.clone().matmul(cols.clone())));
    let cols_t = rnd2([n * l, k]);
    let w_t = rnd2([k, cout]);
    time("same, transposed: [L,K]@[K,Co]", conv_flops, &|| first2(cols_t.clone().matmul(w_t.clone())));
    let x = rnd4([n, 256, H / 4, W / 4]);
    let wt = rnd4([cout, 256, 3, 3]);
    time("same conv, im2col kernel + matmul (conv2d_im2col)", conv_flops, &|| first4(kernels::conv2d_im2col(x.clone(), wt.clone(), None, 1, [1, 1], 120 << 20, false)));
    time("same conv, im2col kernel + kernels::gemm", conv_flops, &|| first4(kernels::conv2d_im2col(x.clone(), wt.clone(), None, 1, [1, 1], 120 << 20, true)));
    time("kernels::gemm [1,Co,K]@[1,K,L]", conv_flops, &|| first(kernels::gemm(w.clone(), cols.clone(), None)));
    let gw = rnd3([1, 384, 640]);
    let gc = rnd3([2, 640, l]);
    time("GRU [1,384,640]@[2,640,L]: burn", 2.0 * (2 * 384 * 640 * l) as f64, &|| first(gw.clone().matmul(gc.clone())));
    time("GRU [1,384,640]@[2,640,L]: kernels::gemm", 2.0 * (2 * 384 * 640 * l) as f64, &|| first(kernels::gemm(gw.clone(), gc.clone(), None)));
    time("same conv, burn conv2d", conv_flops, &|| first4(burn::tensor::module::conv2d(x.clone(), wt.clone(), None, burn::tensor::ops::ConvOptions::new([1, 1], [1, 1], [1, 1], 1))));
    let tokens = 2 * l;
    let a = rnd2([tokens, 256]);
    let fc1 = rnd2([256, 1024]);
    time("FFN fc1 @1/4 x2: [2L,256]@[256,1024]", 2.0 * (tokens * 256 * 1024) as f64, &|| first2(a.clone().matmul(fc1.clone())));
    let a3 = a.clone().unsqueeze_dim::<3>(0);
    let fc1_3 = fc1.clone().unsqueeze_dim::<3>(0);
    time("FFN fc1 @1/4 x2: kernels::gemm", 2.0 * (tokens * 256 * 1024) as f64, &|| first(kernels::gemm(a3.clone(), fc1_3.clone(), None)));
    // Tilings of kernels::gemm (block / k step / outputs per thread per side), on the conv, the GRU and the FFN shapes
    for tile in GEMM_TILES {
        let kernels::GemmTile { block, kstep, per } = tile;
        time(&format!("tile {block}/{kstep}/{per} conv [1,192,2304]@[1,2304,L]"), conv_flops, &|| first(kernels::gemm_tiled(w.clone(), cols.clone(), None, tile)));
        time(&format!("tile {block}/{kstep}/{per} GRU [1,384,640]@[2,640,L]"), 2.0 * (2 * 384 * 640 * l) as f64, &|| first(kernels::gemm_tiled(gw.clone(), gc.clone(), None, tile)));
        time(&format!("tile {block}/{kstep}/{per} FFN [1,2L,256]@[1,256,1024]"), 2.0 * (tokens * 256 * 1024) as f64, &|| first(kernels::gemm_tiled(a3.clone(), fc1_3.clone(), None, tile)));
    }
    let q = rnd3([2 * 64, 180, 128]);
    let kt = rnd3([2 * 64, 128, 180]);
    time("window attention @1/4 x2: [128,180,128]@[128,128,180]", 2.0 * (128 * 180 * 180 * 128) as f64, &|| first(q.clone().matmul(kt.clone())));
    let s = rnd2([2048, 2048]);
    time("square 2048^3", 2.0 * 2048f64.powi(3), &|| first2(s.clone().matmul(s.clone())));
    let like = Tensor::<1>::zeros([1 << 20], &device);
    let peak_flops = (16 * gmflow_dev::bench::FMA_ITERS * (1 << 20)) as f64;
    time("arithmetic peak (independent multiply-add chains)", peak_flops, &|| gmflow_dev::bench::fma_peak(like.clone()).slice([0..1]));
}

/// The bidirectional pass alone, against unimatch with the gate of matches_pytorch: run several times in fresh processes
/// with the autotune cache off (the workflow), to catch a wrong result that depends on which kernels autotune picks
#[test]
#[ignore]
fn bidirectional_repeat() {
    let (device, name) = device();
    println!("device: {name}");
    let dir = env_path("GMFLOW_REF_DIR");
    let model = load(&device);
    let mut st = Statics::new(H, W, &device);
    let im2col = std::env::var("GMFLOW_IM2COL").map(|v| v == "1").unwrap_or(true);
    st.ops = model::Ops { kernels: true, conv_im2col: im2col, gru_factored: true, own_gemm: im2col, gemm_linear: im2col };
    let inp = |k: usize, i: usize| tensor4(&dir.join(format!("pair{k}_in{i}.f32")), [1, 3, H, W], &device);
    let mut bad = Vec::new();
    for k in 0..3 {
        let fa = model.features(inp(k, 0), &st);
        let fb = model.features(inp(k, 1), &st);
        let both = model.flow(&fa, &fb, true, &st, None);
        let fwd = to_vec(both.clone().narrow(0, 0, 1));
        let bwd = to_vec(both.narrow(0, 1, 1));
        for (d, got, file) in [("a -> b", fwd, format!("pair{k}_torch.f32")), ("b -> a", bwd, format!("pair{k}_bwd_torch.f32"))] {
            let (max, mean, _) = diff(&got, &read_f32(&dir.join(&file)));
            println!("REPEAT pair {k} {d}: max |burn - torch| {max:.2e} px, mean {mean:.2e} px");
            if !(max <= 1e-2 && mean <= 1e-4) { bad.push(format!("pair {k} {d}: max {max:e}, mean {mean:e}")); }
        }
    }
    assert!(bad.is_empty(), "bidirectional: {bad:?}");
}
