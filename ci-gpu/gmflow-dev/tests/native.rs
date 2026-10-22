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

/// The configurations measured: (label, ops)
fn configs() -> Vec<(&'static str, model::Ops)> {
    let o = |kernels, conv_im2col, gru_factored| model::Ops { kernels, conv_im2col, gru_factored };
    vec![
        ("plain", o(false, false, false)),
        ("kernels", o(true, false, false)),
        ("kernels+gru", o(true, false, true)),
        ("kernels+im2col", o(true, true, false)),
        ("kernels+im2col+gru", o(true, true, true)),
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
            println!("{label:24} pair {k}: max |burn - torch| {max:.2e} px, mean {mean:.2e} px, one pass {secs:.3} s");
            assert!(max <= 1e-2 && mean <= 1e-4, "pair {k}: max {max:e}, mean {mean:e}");
            checked += 1;
        }
        assert!(checked >= 3, "found {checked} reference pairs");
    }

    // Bidirectional, each frame through the CNN on its own (as the tracker will): the flow both ways against unimatch with
    // pred_bidir_flow, the same gate
    st.ops = model::Ops { kernels: true, conv_im2col: std::env::var("GMFLOW_IM2COL").map(|v| v == "1").unwrap_or(true), gru_factored: true };
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
            println!("bidirectional pair {k} {dir}: max |burn - torch| {max:.2e} px, mean {mean:.2e} px ({secs:.3} s for both, without the CNN)");
            assert!(max <= 1e-2 && mean <= 1e-4, "bidirectional pair {k} {dir}: max {max:e}, mean {mean:e}");
        }
    }

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
        let limit = if label == "local_correlation_softmax" { 1e-3 } else { 1e-4 };
        println!("{label:28}: max |kernel - plain| {max:.2e}, mean {mean:.2e} (values up to {scale:.2e}, limit {limit:e})");
        if !(max <= limit) { failed.push(format!("{label}: {max:e}")); }
    };
    check("local_correlation_with_flow", kernels::corr_with_flow(f0.clone(), f1.clone(), flow.clone()), model::local_correlation_with_flow(f0.clone(), f1.clone(), flow.clone(), kernels::CORR_RADIUS));
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
        let got = kernels::conv2d_im2col(x, wt, Some(bias), stride, [ph, pw], 120 << 20);
        check(&format!("conv {cin}->{cout} {kh}x{kw}/{stride} im2col"), got, want);
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
