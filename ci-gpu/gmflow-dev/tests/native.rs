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
use std::sync::atomic::Ordering;
use std::time::Instant;

const H: usize = 320;
const W: usize = 576;

/// The tilings of gemm_kernel (kernels::gemm_tiled) the ceiling test times (block / k step / outputs per thread per side), the default first;
/// kernels_match_plain checks every one of them against burn's matmul
const GEMM_TILES: [kernels::GemmTile; 5] = [
    kernels::GEMM_DEFAULT,
    kernels::GemmTile { block: 64, kstep: 16, per: 4 },
    kernels::GemmTile { block: 32, kstep: 16, per: 2 },
    kernels::GemmTile { block: 32, kstep: 32, per: 2 },
    kernels::GemmTile { block: 128, kstep: 8, per: 8 },
];

/// The tilings of kernels::gemm4 checked and timed (rows / columns per thread, k step)
const GEMM4_TILES: [kernels::Gemm4Tile; 5] = [
    kernels::GEMM4_DEFAULT,
    kernels::Gemm4Tile { pm: 4, pn: 4, kstep: 16 },
    kernels::Gemm4Tile { pm: 8, pn: 4, kstep: 8 },
    kernels::Gemm4Tile { pm: 4, pn: 8, kstep: 8 },
    kernels::Gemm4Tile { pm: 8, pn: 8, kstep: 8 },
];

/// The configurations measured: (label, ops). The cooperative-matrix one only on a device that offers them (device()
/// says, so it must have been called first)
fn configs() -> Vec<(&'static str, model::Ops)> {
    let o = |kernels, conv_im2col, gru_factored, own_gemm, gemm_linear, fused, implicit| model::Ops { kernels, conv_im2col, gru_factored, own_gemm, gemm_linear, fused, implicit, coop: false };
    let mut all = vec![
        ("plain", o(false, false, false, false, false, false, false)),
        ("kernels", o(true, false, false, false, false, false, false)),
        ("kernels+gru", o(true, false, true, false, false, false, false)),
        ("kernels+gru+fused", o(true, false, true, false, false, true, false)),
        ("kernels+im2col", o(true, true, false, false, false, false, false)),
        ("kernels+im2col+gru", o(true, true, true, false, false, false, false)),
        ("kernels+im2col+gru+gemm", o(true, true, true, true, false, false, false)),
        ("kernels+im2col+gru+gemm+linear", o(true, true, true, true, true, false, false)),
        ("kernels+im2col+gru+gemm+linear+fused", o(true, true, true, true, true, true, false)),
        ("kernels+im2col+gru+gemm+linear+fused+implicit", o(true, true, true, true, true, true, true)),
    ];
    if coop_offered() {
        all.push(("kernels+im2col+gru+gemm+linear+fused+implicit+coop", model::Ops { coop: true, ..o(true, true, true, true, true, true, true) }));
    }
    all
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
    let mut kernels_pair0: Option<Vec<f32>> = None;
    // With cooperative matrices (production where the device has them), the kernels without them (production elsewhere)
    // and the plain ops; the first only on a device that has them
    let mut variants = vec![("kernels", true, false), ("plain ops", false, false)];
    if im2col && coop_offered() {
        variants.insert(0, ("kernels+coop", true, true));
    }
    for (what, kernels_on, coop) in variants {
        st.ops = model::Ops { kernels: kernels_on, conv_im2col: im2col, gru_factored: true, own_gemm: im2col && kernels_on, gemm_linear: im2col && kernels_on, fused: kernels_on, implicit: im2col && kernels_on, coop };
        for k in 0..3 {
            let fa = model.features(inp(k, 0), &st);
            let fb = model.features(inp(k, 1), &st);
            let started = Instant::now();
            let both = model.flow(&fa, &fb, true, &st, None);
            let fwd = to_vec(both.clone().narrow(0, 0, 1));
            let bwd = to_vec(both.narrow(0, 1, 1));
            let secs = started.elapsed().as_secs_f64();
            // The first variant with the kernels: the one the traced passes below run (coop where the device has it)
            if kernels_on && k == 0 && kernels_pair0.is_none() { kernels_pair0 = Some([fwd.clone(), bwd.clone()].concat()); }
            for (dir, got, file) in [("a -> b", fwd, format!("pair{k}_torch.f32")), ("b -> a", bwd, format!("pair{k}_bwd_torch.f32"))] {
                let (max, mean, _) = diff(&got, &read_f32(&env_path("GMFLOW_REF_DIR").join(&file)));
                println!("bidirectional ({what}) pair {k} {dir}: max |burn - torch| {max:.2e} px, mean {mean:.2e} px ({secs:.3} s for both, without the CNN)");
                if !(max <= 1e-2 && mean <= 1e-4) { bad.push(format!("{what} pair {k} {dir}: max {max:e}, mean {mean:e}")); }
            }
        }
    }
    st.ops = model::Ops { kernels: true, conv_im2col: im2col, gru_factored: true, own_gemm: im2col, gemm_linear: im2col, fused: true, implicit: im2col, coop: im2col && coop_offered() };
    // Where a wrong bidirectional pass departs (it failed now and then with the same wrong flow on MSL and WGSL, the plain
    // ops too), run after every gated pass: traced passes keep every stage alive, which changes what burn fuses and does in
    // place. Pair 0 with the kernels: the first item of a traced bidirectional pass against a traced one-way pass of the
    // same features, stage by stage (equal up to rounding when right), and the traced bidirectional pass against the gated
    // untraced one, both items
    {
        let fa = model.features(inp(0, 0), &st);
        let fb = model.features(inp(0, 1), &st);
        let mut t_bi = Trace::default();
        let traced = to_vec(model.flow(&fa, &fb, true, &st, Some(&mut t_bi)));
        let mut t_one = Trace::default();
        let _ = model.flow(&fa, &fb, false, &st, Some(&mut t_one));
        if let Some(untraced) = &kernels_pair0 {
            let (m, _, _) = diff(&traced, untraced);
            println!("BISTAGE traced bidirectional pass against the gated untraced one (both items): max {m:.2e}");
        }
        for (name, tb) in &t_bi.tensors {
            let Some((_, to)) = t_one.tensors.iter().find(|(n, _)| n == name) else { continue };
            let (db, dt) = (tb.dims(), to.dims());
            if db[1..] != dt[1..] || dt[0] != 1 {
                println!("BISTAGE {name}: shapes {db:?} and {dt:?}");
                continue;
            }
            let (m, mean, _) = diff(&to_vec(tb.clone().narrow(0, 0, 1)), &to_vec(to.clone()));
            println!("BISTAGE {name:22}: max |bi[0] - one-way| {m:.2e}, mean {mean:.2e}");
        }
    }
    // Then a one-way pass of pair 0 again (forward: the CNN on both frames and the one-way flow) against PyTorch: wrong if
    // something the passes share (weights, statics) was left wrong; right if only the bidirectional path goes wrong, or
    // if the fault did not last
    {
        let flow = to_vec(model.forward(inp(0, 0), inp(0, 1), &st));
        let (max, mean, _) = diff(&flow, &read_f32(&dir.join("pair0_torch.f32")));
        println!("AFTER the bidirectional passes, one-way pair 0: max |burn - torch| {max:.2e} px, mean {mean:.2e} px");
    }
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
    let mut measured: Vec<(String, model::Ops, bool)> = configs().into_iter().map(|(label, ops)| (label.to_string(), ops, false)).collect();
    // The two own-GEMM configurations again with gemm4 and conv_gemm4 launched unchecked (the same source; on WGSL naga
    // then bounds none of their loops). conv_gemm4 runs only with implicit
    for name in ["kernels+im2col+gru+gemm+linear+fused", "kernels+im2col+gru+gemm+linear+fused+implicit"] {
        let ops = configs().into_iter().find(|(label, _)| *label == name).expect("the configuration").1;
        measured.push((format!("{name}, unchecked"), ops, true));
    }
    for (label, ops, unchecked) in measured {
    st.ops = ops;
    kernels::GEMM4_UNCHECKED.store(unchecked, Ordering::Relaxed);
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
    let sd = (last.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / 5.0).sqrt();
    println!("warm, {label}: {mean:.3} s per pass, sd {sd:.3} (the last 5 of {} passes: {:?})", times.len(), last.iter().map(|t| format!("{t:.3}")).collect::<Vec<_>>());
    }
    kernels::GEMM4_UNCHECKED.store(false, Ordering::Relaxed);
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
        // softmax_flow: weighted sums of 2880 values up to ~50 (pixel coordinates in the model), in another order than
        // burn's softmax then matmul; float32's step at 50 is 3.8e-6, and the sums differ by up to 1.56e-4 (lavapipe and
        // macOS 15 at a2b7087, where the random inputs happened to give 1e-4 and over): 1e-3, far under a wrong weight
        let rounding_of_positions = label == "local_correlation_softmax" || label.contains("corr_with_flow") || label == "local_correlation_with_flow" || label.starts_with("softmax_flow");
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
    check("convex upsampling", kernels::convex_upsample(flow.clone(), mask.clone()), model::convex_upsample(flow.clone(), mask));
    let flow8 = rnd([b, 2, H / 8, W / 8], 3.0);
    check("flow doubled", kernels::flow_double(flow8.clone()), model::flow_double(flow8));
    // The flow head's last convolution (256 -> 2, 3x3) by conv_small, against burn
    {
        let x = rnd([b, 256, h, w], 1.0);
        let wt = rnd([2, 256, 3, 3], 0.02);
        let bias = Tensor::<1>::random([2], Distribution::Normal(0.0, 0.1), &device);
        let want = burn::tensor::module::conv2d(x.clone(), wt.clone(), Some(bias.clone()), burn::tensor::ops::ConvOptions::new([1, 1], [1, 1], [1, 1], 1));
        check("conv_small 256->2 3x3", kernels::conv_small(x, wt, bias, 1, 1), want);
    }
    // Split-window attention at both sizes of the model, shifted and not, at the one-way and bidirectional batches (f0
    // and f1 concatenated: 2 and 4), against the plain operations
    for (hh, ww, splits) in [(H / 4, W / 4, 8usize), (H / 8, W / 8, 2)] {
        let win = model::Windows::new(hh, ww, splits, &device);
        for bn in [2usize, 4] {
            let r3 = |s: [usize; 3]| Tensor::<3>::random(s, Distribution::Normal(0.0, 1.0), &device);
            let (q, k, v) = (r3([bn, hh * ww, c]), r3([bn, hh * ww, c]), r3([bn, hh * ww, c]));
            for shift in [false, true] {
                let got = kernels::window_attention(q.clone(), k.clone(), v.clone(), hh, ww, splits, shift);
                let want = model::split_window_attention(q.clone(), k.clone(), v.clone(), &win, shift, false, false);
                check(&format!("window attention {hh}x{ww}/{splits} b{bn} shift {shift}"), got.unsqueeze_dim::<4>(0), want.clone().unsqueeze_dim::<4>(0));
                // The same with its matmuls on cooperative matrices (coop.rs), where the device has them
                if coop_offered() {
                    let got = gmflow_dev::coop::window_attention(q.clone(), k.clone(), v.clone(), hh, ww, splits, shift, gmflow_dev::coop::wgsl::COOP_DEFAULT);
                    check(&format!("window attention {hh}x{ww}/{splits} b{bn} shift {shift}, coop"), got.unsqueeze_dim::<4>(0), want.unsqueeze_dim::<4>(0));
                }
            }
        }
    }
    // The transformer's LayerNorms (+ residual) by layer_norm_kernel, against burn's LayerNorm with its default epsilon
    // (the model's) and an add: 128 channels at both sizes and batches of the model, and 37 channels in 7 rows (lanes
    // and rows past the end of the cubes)
    for (rows, ch) in [(2 * (H / 4) * (W / 4), c), (4 * (H / 8) * (W / 8), c), (7, 37)] {
        let r3 = |sd: f64| Tensor::<3>::random([1, rows, ch], Distribution::Normal(0.0, sd), &device);
        let (x, res) = (r3(2.0).add_scalar(0.5), r3(1.0));
        let mut ln = burn::nn::LayerNormConfig::new(ch).init(&device);
        ln.gamma = burn::module::Param::from_tensor(Tensor::<1>::random([ch], Distribution::Normal(1.0, 0.2), &device));
        ln.beta = Some(burn::module::Param::from_tensor(Tensor::<1>::random([ch], Distribution::Normal(0.0, 0.2), &device)));
        let (gamma, beta) = (ln.gamma.val(), ln.beta.as_ref().map(|b| b.val()).unwrap());
        let want = ln.forward(x.clone());
        let got = kernels::layer_norm(x.clone(), gamma.clone(), beta.clone(), None);
        check(&format!("layer norm {rows}x{ch}"), got.unsqueeze_dim::<4>(0), want.clone().unsqueeze_dim::<4>(0));
        let got = kernels::layer_norm(x, gamma, beta, Some(res.clone()));
        check(&format!("layer norm + residual {rows}x{ch}"), got.unsqueeze_dim::<4>(0), (res + want).unsqueeze_dim::<4>(0));
    }
    // The CNN's InstanceNorms by instance_norm_kernel, against burn's InstanceNorm (no affine, eps 1e-5 as the model's)
    // and the activations after it: alone (downsample), with relu, with relu and the residual block's relu(res + y). The
    // CNN's first (64 channels at 1/2) and a block's at 1/4 (96), 5 channels of 7x37 (a row not a multiple of the cube's
    // 256 threads) and of 5x7 (fewer pixels than threads); values off zero so that the mean matters, and in the last
    // shape of sd 1e-3, a variance under eps, so that eps and where it is added matter
    {
        use burn::tensor::activation::relu;
        for (shape, sd) in [([2, 64, H / 2, W / 2], 2.0), ([2, 96, H / 4, W / 4], 2.0), ([1, 5, 7, 37], 2.0), ([1, 5, 5, 7], 2.0), ([1, 5, 7, 37], 1e-3)] {
            let x = rnd(shape, sd).add_scalar(3.0 * sd);
            let res = rnd(shape, 1.0);
            let norm = burn::nn::InstanceNormConfig::new(shape[1]).with_affine(false).with_epsilon(1e-5).init(&device);
            let y = norm.forward(x.clone());
            let label = format!("{shape:?} sd {sd}");
            check(&format!("instance norm {label}"), kernels::instance_norm(x.clone(), false, None), y.clone());
            check(&format!("instance norm + relu {label}"), kernels::instance_norm(x.clone(), true, None), relu(y.clone()));
            check(&format!("instance norm + relu + residual {label}"), kernels::instance_norm(x, true, Some(res.clone())), relu(res + relu(y)));
        }
    }
    // Global matching and global propagation at 1/8 by softmax_flow_kernel (the softmax over the keys and its product with 2
    // columns in one launch), against burn's softmax and matmul: the bidirectional pass's sizes with the 2 columns shared
    // by the batch (the grid of positions) and per batch (the flow), scores of sd 3 (a peaked softmax, as the correlation
    // is); 7 rows of 37 keys (fewer keys than threads) and of 300 (a cube's keys read in two rounds) with a batch of 3
    {
        use burn::tensor::activation::softmax;
        let l8 = (H / 8) * (W / 8);
        for (bn, n, l, vb, sd) in [(2usize, l8, l8, 1usize, 3.0f64), (2, l8, l8, 2, 3.0), (1, 7, 37, 1, 1.0), (3, 7, 300, 3, 10.0)] {
            let scores = Tensor::<3>::random([bn, n, l], Distribution::Normal(0.0, sd), &device);
            let v = Tensor::<3>::random([vb, l, 2], Distribution::Normal(0.0, 20.0), &device);
            let want = softmax(scores.clone(), 2).matmul(if vb == 1 { v.clone().repeat_dim(0, bn) } else { v.clone() });
            check(&format!("softmax_flow {bn}x{n}x{l} v{vb} sd{sd}"), kernels::softmax_flow(scores, v).unsqueeze_dim::<4>(0), want.unsqueeze_dim::<4>(0));
        }
    }
    // The FFN's second matmul with the GELU of its input taken as coop.rs reads it (coop::gemm_gelu_a), against burn's
    // GELU then the same matmul: the bidirectional pass's 1/4 tokens of 1024 channels (vec4 loads of a), and 37 rows of 13
    // (scalar loads, every edge); inputs of sd 2, so that the GELU's negative side and its tail both matter
    if coop_offered() {
        use burn::tensor::activation::gelu;
        use gmflow_dev::coop;
        for (rows, k, n) in [(2 * (H / 4) * (W / 4), 1024usize, 128usize), (37, 13, 70)] {
            let a = Tensor::<3>::random([1, rows, k], Distribution::Normal(0.0, 2.0), &device);
            let b = Tensor::<3>::random([1, k, n], Distribution::Normal(0.0, 0.05), &device);
            let want = coop::gemm(gelu(a.clone()), b.clone(), coop::wgsl::COOP_DEFAULT);
            let got = coop::gemm_gelu_a(a, b, coop::wgsl::COOP_DEFAULT);
            check(&format!("coop gemm, gelu of a {rows}x{k}x{n}"), got.unsqueeze_dim::<4>(0), want.unsqueeze_dim::<4>(0));
        }
        // The FFN's first matmul on its input and the message side by side (coop::gemm_cat_a), against the same matmul of
        // their concatenation: the 1/4 tokens of 128 + 128 channels into 1024 (vec4 loads), 37 rows of 32 + 12 (vec4,
        // the last k step partial) and of 16 + 13 (scalar loads, every edge)
        for (rows, k1, k2, n) in [(2 * (H / 4) * (W / 4), 128usize, 128usize, 1024usize), (37, 32, 12, 70), (37, 16, 13, 70)] {
            let a = Tensor::<3>::random([1, rows, k1], Distribution::Normal(0.0, 1.0), &device);
            let a2 = Tensor::<3>::random([1, rows, k2], Distribution::Normal(0.0, 1.0), &device);
            let b = Tensor::<3>::random([1, k1 + k2, n], Distribution::Normal(0.0, 0.1), &device);
            let want = coop::gemm(Tensor::cat(vec![a.clone(), a2.clone()], 2), b.clone(), coop::wgsl::COOP_DEFAULT);
            let got = coop::gemm_cat_a(a, a2, b, coop::wgsl::COOP_DEFAULT);
            check(&format!("coop gemm, a in two {rows}x({k1}+{k2})x{n}"), got.unsqueeze_dim::<4>(0), want.unsqueeze_dim::<4>(0));
        }
        // The bias and relu taken as coop.rs writes a convolution (coop::conv2d_act), against coop.rs's convolution without
        // them, then burn's add and relu: the motion encoder's 3x3 (256 -> 192 at 1/4, batch 2) and its 1x1 (81 -> 256, a
        // matmul), and 37 -> 45 on 37 x 53 (every edge) with a relu and no bias
        use burn::tensor::activation::relu;
        let (h4, w4) = (H / 4, W / 4);
        for (label, nb, ci, co, hh, ww, kk, with_bias) in [("3x3 256->192", 2usize, 256usize, 192usize, h4, w4, 3usize, true), ("1x1 81->256", 2, 81, 256, h4, w4, 1, true), ("3x3 37->45 no bias", 1, 37, 45, 37, 53, 3, false)] {
            let x = rnd([nb, ci, hh, ww], 1.0);
            let wt = rnd([co, ci, kk, kk], 0.05);
            let bias = with_bias.then(|| Tensor::<1>::random([co], Distribution::Normal(0.0, 0.5), &device));
            let pad = [kk / 2, kk / 2];
            let plain = coop::conv2d(x.clone(), wt.clone(), None, 1, pad, coop::wgsl::COOP_DEFAULT);
            let plain = match &bias { Some(b) => plain + b.clone().reshape([1, co, 1, 1]), None => plain };
            let got = coop::conv2d_act(x, wt, bias, 1, pad, true, coop::wgsl::COOP_DEFAULT);
            check(&format!("coop conv + bias + relu {label}"), got, relu(plain));
        }
    }
    // The factored GRU's gates by gru_gate_kernel against the plain operations of GruFactored::step (burn's sigmoid and
    // tanh), in the layouts of its two halves: 2 z/r terms and q from an extra tensor and a z/r term's (horizontal), 3
    // z/r terms and q from two of theirs (vertical). Pre-activations of sd 3 so that the sigmoid and tanh saturate, the
    // GRU's 128 channels, a position count not a multiple of the cube
    {
        use burn::tensor::activation::{ sigmoid, tanh };
        use kernels::QFrom;
        let l = (H / 8) * (W / 8) + 3;
        let r3 = |ch: usize, sd: f64| Tensor::<3>::random([2, ch, l], Distribution::Normal(0.0, sd), &device);
        let (za, zb, zc, h, qa) = (r3(256, 3.0), r3(384, 3.0), r3(384, 3.0), r3(128, 1.0), r3(128, 3.0));
        for three in [false, true] {
            let (terms, zr, q_plain, q_terms) = if three {
                let zr = za.clone() + zb.clone().narrow(1, 0, 256) + zc.clone().narrow(1, 0, 256);
                let q = qa.clone() + zb.clone().narrow(1, 256, 128) + zc.clone().narrow(1, 256, 128);
                (vec![(za.clone(), 0), (zb.clone(), 0), (zc.clone(), 0)], zr, q, ((qa.clone(), 0), None, [(QFrom::Zr(1), 256), (QFrom::Zr(2), 256)]))
            } else {
                // The extra tensor first, as the horizontal half's c.q1
                let zr = za.clone() + zb.clone().narrow(1, 0, 256);
                let q = qa.clone() + zc.clone().narrow(1, 256, 128) + zb.clone().narrow(1, 256, 128);
                (vec![(za.clone(), 0), (zb.clone(), 0)], zr, q, ((qa.clone(), 0), Some(zc.clone()), [(QFrom::Extra, 256), (QFrom::Zr(1), 256)]))
            };
            let z = sigmoid(zr.clone().narrow(1, 0, 128));
            let r = sigmoid(zr.narrow(1, 128, 128));
            let want_reset = r * h.clone();
            let want_update = z.clone().neg().add_scalar(1.0) * h.clone() + z * tanh(q_plain);
            let got = kernels::gru_gate(&terms, None, h.clone());
            check(&format!("gru reset, {} terms", terms.len()), got.unsqueeze_dim::<4>(0), want_reset.unsqueeze_dim::<4>(0));
            let got = kernels::gru_gate(&terms, Some(q_terms), h.clone());
            check(&format!("gru update, {} terms", terms.len()), got.unsqueeze_dim::<4>(0), want_update.unsqueeze_dim::<4>(0));
        }
    }
    // Convolutions of the shapes of the model as im2col + matmul, against burn
    for (cin, cout, kh, kw, stride, ph, pw, hh, ww) in [(3usize, 64usize, 7usize, 7usize, 2usize, 3usize, 3usize, H, W), (64, 64, 3, 3, 1, 1, 1, H / 2, W / 2), (64, 96, 3, 3, 2, 1, 1, H / 2, W / 2), (256, 192, 3, 3, 1, 1, 1, h, w), (384, 128, 1, 5, 1, 0, 2, h, w), (384, 128, 5, 1, 1, 2, 0, h, w)] {
        let x = rnd([2, cin, hh, ww], 1.0);
        let wt = rnd([cout, cin, kh, kw], 0.02); // outputs of order 1, as the other checks
        let bias = Tensor::<1>::random([cout], Distribution::Normal(0.0, 0.1), &device);
        let want = burn::tensor::module::conv2d(x.clone(), wt.clone(), Some(bias.clone()), burn::tensor::ops::ConvOptions::new([stride, stride], [ph, pw], [1, 1], 1));
        let got = kernels::conv2d_im2col(x.clone(), wt.clone(), Some(bias.clone()), stride, [ph, pw], 120 << 20, false);
        check(&format!("conv {cin}->{cout} {kh}x{kw}/{stride} im2col"), got, want.clone());
        let got = kernels::conv2d_im2col(x.clone(), wt.clone(), Some(bias.clone()), stride, [ph, pw], 120 << 20, true);
        check(&format!("conv {cin}->{cout} {kh}x{kw}/{stride} im2col gemm"), got, want.clone());
        let got = kernels::conv2d_implicit(x, wt, Some(bias), stride, [ph, pw]);
        check(&format!("conv {cin}->{cout} {kh}x{kw}/{stride} implicit"), got, want);
    }
    // conv_gemm (columns gathered in the kernel) against gemm on the im2col columns, at the GRU's 1x5 and 5x1 and the
    // trident's stride 2, at a size whose OH*OW is not a multiple of 4 (im2col + gemm), and at output widths not a
    // multiple of 4 (vectors of 4 positions across two output rows)
    for (cin, cout, kh, kw, stride, ph, pw, hh, ww) in [(128usize, 384usize, 1usize, 5usize, 1usize, 0usize, 2usize, h, w), (128, 384, 5, 1, 1, 2, 0, h, w), (128, 128, 3, 3, 2, 1, 1, h, w), (16, 64, 3, 3, 1, 1, 1, 7, 9), (16, 64, 3, 3, 1, 1, 1, 6, 10), (16, 64, 3, 3, 2, 1, 1, 8, 14)] {
        let x = rnd([2, cin, hh, ww], 1.0);
        let wm = Tensor::<3>::random([1, cout, cin * kh * kw], Distribution::Normal(0.0, 1.0 / ((cin * kh * kw) as f64).sqrt()), &device);
        let want = kernels::gemm(wm.clone(), kernels::im2col(x.clone(), kh, kw, stride, ph, pw), None).unsqueeze_dim::<4>(0);
        check(&format!("conv_gemm {cin}->{cout} {kh}x{kw}/{stride} {hh}x{ww}"), kernels::conv_gemm(wm, x, kh, kw, stride, ph, pw).unsqueeze_dim::<4>(0), want);
    }
    // kernels::gemm against burn's matmul, with every tiling the ceiling test times (the default first): the shapes of the
    // model (a of 1 shared by the batch) and both batched, some with rows and columns not multiples of the 32, 64 or 128
    // block and k not a multiple of the 8, 16 or 32 step (every tiling meets partial tiles in all three); a scaled by 1/sqrt(K) so that the outputs are of order 1
    let rnd3 = |shape: [usize; 3], sd: f64| Tensor::<3>::random(shape, Distribution::Normal(0.0, sd), &device);
    assert_eq!(GEMM_TILES[0], kernels::GEMM_DEFAULT, "the default tiling comes first");
    assert_eq!(GEMM4_TILES[0], kernels::GEMM4_DEFAULT, "the default gemm4 tiling comes first");
    for (ba, bb, m, k, nn) in [(1usize, 1usize, 192usize, 2304usize, 11520usize), (1, 2, 126, 1152, 11520), (1, 2, 384, 640, 11520), (1, 1, 64, 147, 46080), (2, 2, 180, 128, 180), (1, 2, 2, 1152, 11520), (2, 1, 100, 256, 500)] {
        let a = rnd3([ba, m, k], 1.0 / (k as f64).sqrt());
        let bm = rnd3([bb, k, nn], 1.0);
        let want = a.clone().matmul(bm.clone()).unsqueeze_dim::<4>(0);
        check(&format!("gemm [{ba},{m},{k}]x[{bb},{k},{nn}]"), kernels::gemm(a.clone(), bm.clone(), None).unsqueeze_dim::<4>(0), want.clone());
        // gemm is gemm4 (or gemm_tiled with GEMM_TILES[0] where N is not a multiple of 4): every gemm_kernel tiling
        for tile in &GEMM_TILES {
            let tile = *tile;
            let label = format!("gemm {}/{}/{} [{ba},{m},{k}]x[{bb},{k},{nn}]", tile.block, tile.kstep, tile.per);
            check(&label, kernels::gemm_tiled(a.clone(), bm.clone(), None, tile).unsqueeze_dim::<4>(0), want.clone());
        }
        for tile in GEMM4_TILES {
            let label = format!("gemm4 {}x{}/{} [{ba},{m},{k}]x[{bb},{k},{nn}]", tile.pm, tile.pn, tile.kstep);
            check(&label, kernels::gemm4_tiled(a.clone(), bm.clone(), None, tile).unsqueeze_dim::<4>(0), want.clone());
        }
    }
    {
        let a = rnd3([1, 100, 300], 0.06);
        let bm = rnd3([2, 300, 333], 1.0);
        let bias = Tensor::<1>::random([100], Distribution::Normal(0.0, 0.1), &device);
        let want = (a.clone().matmul(bm.clone()) + bias.clone().reshape([1, 100, 1])).unsqueeze_dim::<4>(0);
        check("gemm with bias [1,100,300]x[2,300,333]", kernels::gemm(a.clone(), bm.clone(), Some(bias.clone())).unsqueeze_dim::<4>(0), want.clone());
        for tile in &GEMM_TILES {
            let tile = *tile;
            let label = format!("gemm {}/{}/{} with bias [1,100,300]x[2,300,333]", tile.block, tile.kstep, tile.per);
            check(&label, kernels::gemm_tiled(a.clone(), bm.clone(), Some(bias.clone()), tile).unsqueeze_dim::<4>(0), want.clone());
        }
        for tile in GEMM4_TILES {
            let label = format!("gemm4 {}x{}/{} with bias [1,100,300]x[2,300,333] (gemm)", tile.pm, tile.pn, tile.kstep);
            check(&label, kernels::gemm4_tiled(a.clone(), bm.clone(), Some(bias.clone()), tile).unsqueeze_dim::<4>(0), want.clone());
        }
        // gemm4 with a bias, columns a multiple of 4 (333 falls back to gemm)
        let b4 = rnd3([2, 300, 332], 1.0);
        let want4 = (a.clone().matmul(b4.clone()) + bias.clone().reshape([1, 100, 1])).unsqueeze_dim::<4>(0);
        for tile in GEMM4_TILES {
            let label = format!("gemm4 {}x{}/{} with bias [1,100,300]x[2,300,332]", tile.pm, tile.pn, tile.kstep);
            check(&label, kernels::gemm4_tiled(a.clone(), b4.clone(), Some(bias.clone()), tile).unsqueeze_dim::<4>(0), want4.clone());
        }
    }
    assert!(failed.is_empty(), "kernels that differ from the plain operations: {failed:?}");
}

/// burn's bilinear interpolation (the flow doubled at 1/8, aligned corners, as model.rs makes it) with each of cubek's two
/// strategies and with autotune's choice, at batches 1, 2 and 3, against the CPU. Fails only if maximize_throughput is
/// wrong; minimize_latency and autotune's choice are reported (see interp_diag.rs)
#[test]
#[ignore]
fn interpolate_strategies() {
    let (device, name) = device();
    println!("device: {name}");
    let (c, h, w) = (2usize, 40usize, 72usize);
    let (oh, ow) = (2 * h, 2 * w);
    let per_item = c * oh * ow;
    let mut bad = Vec::new();
    for b in [1usize, 2, 3] {
        let x = Tensor::<4>::random([b, c, h, w], Distribution::Normal(0.0, 3.0), &device);
        let xs = to_vec(x.clone());
        let (sy, sx) = ((h - 1) as f64 / (oh - 1) as f64, (w - 1) as f64 / (ow - 1) as f64);
        let mut want = vec![0f32; b * per_item];
        for n in 0..b * c {
            let at = |y: usize, x: usize| xs[(n * h + y) * w + x] as f64;
            for oy in 0..oh {
                let fy = oy as f64 * sy;
                let y0 = (fy.floor() as usize).min(h - 1);
                let (y1, ty) = ((y0 + 1).min(h - 1), fy - y0 as f64);
                for ox in 0..ow {
                    let fx = ox as f64 * sx;
                    let x0 = (fx.floor() as usize).min(w - 1);
                    let (x1, tx) = ((x0 + 1).min(w - 1), fx - x0 as f64);
                    let v = (1.0 - ty) * ((1.0 - tx) * at(y0, x0) + tx * at(y0, x1)) + ty * ((1.0 - tx) * at(y1, x0) + tx * at(y1, x1));
                    want[(n * oh + oy) * ow + ox] = v as f32;
                }
            }
        }
        // Each forced strategy 10 times, each call just after four tensors of the output's size filled with 1e6 were made
        // and dropped (four, as the call first takes the NHWC copy of its input from the pool, then its output): a value the
        // kernel leaves unwritten shows as 1e6. minimize_latency first: the first version of this test ran it right after
        // maximize_throughput, whose correct output was what an unwritten value would have kept
        let mut first: Vec<Vec<f32>> = Vec::new();
        for (label, latency) in [("minimize_latency", true), ("maximize_throughput", false)] {
            let (mut worst, mut worst_mean, mut worst_items, mut stale) = (0f32, 0f64, Vec::new(), 0usize);
            for rep in 0..10 {
                let junk: Vec<Tensor<4>> = (0..4).map(|_| Tensor::<4>::full([b, c, oh, ow], 1.0e6, &device)).collect();
                for t in &junk { let _ = to_vec(t.clone()); }
                drop(junk);
                let got = to_vec(interp_diag::interp2x(x.clone(), latency));
                stale = stale.max(got.iter().filter(|v| **v == 1.0e6).count());
                let (max, mean, _) = diff(&got, &want);
                if rep == 0 || max > worst {
                    worst = max;
                    worst_mean = mean;
                    worst_items = (0..b).map(|i| format!("{:.1e}", diff(&got[i * per_item..(i + 1) * per_item], &want[i * per_item..(i + 1) * per_item]).0)).collect::<Vec<_>>();
                }
                if rep == 0 { first.push(got); }
            }
            println!("INTERP {label:20} batch {b}: worst of 10, max |gpu - cpu| {worst:.2e}, mean {worst_mean:.2e}, max per item {worst_items:?}, most values left at 1e6 {stale}");
            if !(worst <= 1e-4) { bad.push(format!("{label} batch {b}: {worst:e}")); }
        }
        // burn's own call, as the model makes it; which strategy autotune chose shows as an exact match with that one's output
        let got = to_vec(burn::tensor::module::interpolate(x.clone(), burn::tensor::ops::InterpolateOptions::new(burn::tensor::ops::InterpolateMode::Bilinear).with_align_corners(true).with_scale_factor([2.0, 2.0])));
        let (max, mean, _) = diff(&got, &want);
        let same = |o: &Vec<f32>| got.iter().zip(o).all(|(a, b)| a.to_bits() == b.to_bits());
        let chose = match (same(&first[1]), same(&first[0])) { (true, true) => "both (identical outputs)", (true, false) => "maximize_throughput", (false, true) => "minimize_latency", _ => "neither" };
        println!("INTERP {:20} batch {b}: max |gpu - cpu| {max:.2e}, mean {mean:.2e}, output equal to {chose}", "autotune");
        if !(max <= 1e-4) { bad.push(format!("autotune batch {b}: {max:e}")); }
        // What the model uses instead (2 * the interpolation): the kernel and the plain form, against the CPU
        let want2: Vec<f32> = want.iter().map(|v| 2.0 * v).collect();
        for (label, got) in [("kernels::flow_double", kernels::flow_double(x.clone())), ("model::flow_double", model::flow_double(x.clone()))] {
            let (max, mean, _) = diff(&to_vec(got), &want2);
            println!("INTERP {label:20} batch {b}: max |gpu - 2 cpu| {max:.2e}, mean {mean:.2e}");
            if !(max <= 2e-4) { bad.push(format!("{label} batch {b}: {max:e}")); }
        }
    }
    println!("INTERP over 1e-4: {bad:?}");
    assert!(!bad.iter().any(|s| s.starts_with("maximize_throughput") || s.contains("flow_double")), "maximize_throughput or flow_double: {bad:?}");
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
        let mut totals = Vec::new();
        for (start, marks) in &runs {
            let mut prev = *start;
            for (n, t) in marks {
                let l = label(n);
                if !order.contains(&l) { order.push(l.clone()); }
                *sums.entry(l).or_default() += (*t - prev).as_secs_f64();
                prev = *t;
            }
            total += (prev - *start).as_secs_f64();
            totals.push(format!("{:.3}", (prev - *start).as_secs_f64()));
        }
        let k = runs.len() as f64;
        println!("{what} ({want}): {:.3} s per pass with a wait at each stage (mean of {}: {totals:?})", total / k, runs.len());
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
    time("same conv, implicit gemm (conv2d_implicit)", conv_flops, &|| first4(kernels::conv2d_implicit(x.clone(), wt.clone(), None, 1, [1, 1])));
    // The same two launched unchecked (no clamp of each buffer index by cubecl)
    kernels::GEMM4_UNCHECKED.store(true, Ordering::Relaxed);
    time("same conv, im2col kernel + kernels::gemm, unchecked", conv_flops, &|| first4(kernels::conv2d_im2col(x.clone(), wt.clone(), None, 1, [1, 1], 120 << 20, true)));
    time("same conv, implicit gemm (conv2d_implicit), unchecked", conv_flops, &|| first4(kernels::conv2d_implicit(x.clone(), wt.clone(), None, 1, [1, 1])));
    kernels::GEMM4_UNCHECKED.store(false, Ordering::Relaxed);
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
    for tile in GEMM4_TILES {
        let kernels::Gemm4Tile { pm, pn, kstep } = tile;
        time(&format!("gemm4 {pm}x{pn}/{kstep} conv [1,192,2304]@[1,2304,L]"), conv_flops, &|| first(kernels::gemm4_tiled(w.clone(), cols.clone(), None, tile)));
        time(&format!("gemm4 {pm}x{pn}/{kstep} GRU [1,384,640]@[2,640,L]"), 2.0 * (2 * 384 * 640 * l) as f64, &|| first(kernels::gemm4_tiled(gw.clone(), gc.clone(), None, tile)));
        time(&format!("gemm4 {pm}x{pn}/{kstep} FFN [1,2L,256]@[1,256,1024]"), 2.0 * (tokens * 256 * 1024) as f64, &|| first(kernels::gemm4_tiled(a3.clone(), fc1_3.clone(), None, tile)));
    }
    kernels::GEMM4_UNCHECKED.store(true, Ordering::Relaxed);
    {
        let tile = kernels::GEMM4_DEFAULT;
        let kernels::Gemm4Tile { pm, pn, kstep } = tile;
        time(&format!("gemm4 {pm}x{pn}/{kstep} unchecked conv [1,192,2304]@[1,2304,L]"), conv_flops, &|| first(kernels::gemm4_tiled(w.clone(), cols.clone(), None, tile)));
        time(&format!("gemm4 {pm}x{pn}/{kstep} unchecked GRU [1,384,640]@[2,640,L]"), 2.0 * (2 * 384 * 640 * l) as f64, &|| first(kernels::gemm4_tiled(gw.clone(), gc.clone(), None, tile)));
        time(&format!("gemm4 {pm}x{pn}/{kstep} unchecked FFN [1,2L,256]@[1,256,1024]"), 2.0 * (tokens * 256 * 1024) as f64, &|| first(kernels::gemm4_tiled(a3.clone(), fc1_3.clone(), None, tile)));
    }
    kernels::GEMM4_UNCHECKED.store(false, Ordering::Relaxed);
    let q = rnd3([2 * 64, 180, 128]);
    let kt = rnd3([2 * 64, 128, 180]);
    time("window attention @1/4 x2: [128,180,128]@[128,128,180]", 2.0 * (128 * 180 * 180 * 128) as f64, &|| first(q.clone().matmul(kt.clone())));
    // The whole attention of a shifted layer at 1/4 (both matmuls, mask, softmax, window layout), as the model runs it
    {
        let (hh, ww) = (H / 4, W / 4);
        let win = model::Windows::new(hh, ww, 8, &device);
        let (aq, ak, av) = (rnd3([2, hh * ww, 128]), rnd3([2, hh * ww, 128]), rnd3([2, hh * ww, 128]));
        let fl = 2.0 * 2.0 * (128 * 180 * 180 * 128) as f64;
        time("window attention @1/4 x2, shifted, whole: plain ops", fl, &|| first(model::split_window_attention(aq.clone(), ak.clone(), av.clone(), &win, true, false, false)));
        time("window attention @1/4 x2, shifted, whole: kernels", fl, &|| first(kernels::window_attention(aq.clone(), ak.clone(), av.clone(), hh, ww, 8, true)));
        if coop_offered() {
            time("window attention @1/4 x2, shifted, whole: coop", fl, &|| first(gmflow_dev::coop::window_attention(aq.clone(), ak.clone(), av.clone(), hh, ww, 8, true, gmflow_dev::coop::wgsl::COOP_DEFAULT)));
        }
    }
    let s = rnd2([2048, 2048]);
    time("square 2048^3", 2.0 * 2048f64.powi(3), &|| first2(s.clone().matmul(s.clone())));
    let like = Tensor::<1>::zeros([1 << 20], &device);
    let peak_flops = (16 * gmflow_dev::bench::FMA_ITERS * (1 << 20)) as f64;
    time("arithmetic peak (independent multiply-add chains)", peak_flops, &|| gmflow_dev::bench::fma_peak(like.clone()).slice([0..1]));
}

/// Compiles conv_gemm4_kernel and gemm4_kernel (the model's 3x3, stride 1, default tiling) checked and unchecked, and
/// im2col_kernel (always checked). Checks the checked results against burn's conv2d and the unchecked against the
/// checked. With CUBECL_DEBUG_LOG=<file> cubecl writes each compiled source there: the workflow keeps it (sources.log),
/// to compare what WGSL and MSL make of the same kernel. 32 output channels: conv2d_im2col uses gemm from 32 up
#[test]
#[ignore]
fn kernel_sources() {
    let (device, name) = device();
    println!("device: {name}");
    let x = Tensor::<4>::random([1, 16, 8, 16], Distribution::Normal(0.0, 1.0), &device);
    let wt = Tensor::<4>::random([32, 16, 3, 3], Distribution::Normal(0.0, 1.0), &device).mul_scalar(0.02);
    let run = |unchecked: bool| {
        kernels::GEMM4_UNCHECKED.store(unchecked, Ordering::Relaxed);
        let implicit = to_vec(kernels::conv2d_implicit(x.clone(), wt.clone(), None, 1, [1, 1]));
        let im2col = to_vec(kernels::conv2d_im2col(x.clone(), wt.clone(), None, 1, [1, 1], 120 << 20, true));
        kernels::GEMM4_UNCHECKED.store(false, Ordering::Relaxed);
        (implicit, im2col)
    };
    let (ci, cm) = run(false);
    let (ui, um) = run(true);
    let options = burn::tensor::ops::ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
    let want = to_vec(burn::tensor::module::conv2d(x.clone(), wt.clone(), None, options));
    // Checked against burn within rounding; unchecked against checked equal (the same arithmetic in the same order)
    let mut failed = Vec::new();
    let rows = [
        ("implicit vs burn conv2d", &ci, &want, 1e-4f32),
        ("im2col + gemm vs burn conv2d", &cm, &want, 1e-4f32),
        ("implicit unchecked vs checked", &ui, &ci, 0.0f32),
        ("im2col + gemm unchecked vs checked", &um, &cm, 0.0f32),
    ];
    for (label, got, base, limit) in rows {
        let (max, mean, _) = diff(got, base);
        println!("{label:36}: max {max:.2e}, mean {mean:.2e} (limit {limit:.0e})");
        if max > limit {
            failed.push(format!("{label}: max {max:e} over {limit:e}"));
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// What naga makes of conv_gemm4_kernel and gemm4_kernel on Metal: the WGSL that kernel_sources logged (sources.log)
/// translated to MSL with the options wgpu-hal 30 uses for cubecl's modules (cubecl-wgpu: no runtime checks except loop
/// bounding when checked, workgroup memory not zeroed; MSL 3.1, the version wgpu-hal picks on macOS 14; binding i to
/// buffer i, then the sizes buffer), to compare with the MSL cubecl writes itself (sources.log of the metal-msl job).
/// wgpu-hal then compiles with MSL 3.1 and invariance preserved, cubecl's MSL with Metal's default options. No GPU
#[test]
#[ignore]
fn naga_msl() {
    use naga::back::msl;
    let log = std::fs::read_to_string("sources.log").expect("sources.log, from kernel_sources with CUBECL_DEBUG_LOG");
    let mut written = Vec::new();
    for block in log.split("[END_KERNEL_COMPILATION]") {
        let Some(name) = block.lines().find_map(|l| l.strip_prefix("name: gmflow_dev::kernels::")) else { continue };
        let kernel = name.split("::").next().unwrap_or_default();
        if kernel != "conv_gemm4_kernel" && kernel != "gemm4_kernel" {
            continue;
        }
        let mode = block.lines().find_map(|l| l.trim().strip_prefix("mode: ")).expect("the execution mode").trim_end_matches(',');
        let start = block.find("```wgsl\n").expect("the WGSL source") + "```wgsl\n".len();
        let src = &block[start..];
        let src = &src[..src.find("\n```").expect("the end of the source")];
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{kernel} {mode}: {}", e.emit_to_string(src)));
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{kernel} {mode}: {e:?}"));
        // The bindings as wgpu-hal maps cubecl's layout (one group, binding i at layout entry i): buffer i, then the sizes
        // buffer, as every binding is a storage buffer or the info
        let mut resources = msl::BindingMap::new();
        for (_, var) in module.global_variables.iter() {
            if let Some(binding) = &var.binding {
                assert_eq!(binding.group, 0, "{kernel} {mode}: one bind group");
                let mutable = matches!(var.space, naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE));
                resources.insert(binding.clone(), msl::BindTarget { buffer: Some(binding.binding as u8), mutable, ..Default::default() });
            }
        }
        let count = resources.keys().map(|b| b.binding + 1).max().expect("bindings");
        assert_eq!(count as usize, resources.len(), "{kernel} {mode}: bindings 0..{count}");
        let entry = msl::EntryPointResources { resources, sizes_buffer: Some(count as u8), ..Default::default() };
        let options = msl::Options {
            lang_version: (3, 1),
            fake_missing_bindings: false,
            per_entry_point_map: module.entry_points.iter().map(|ep| (ep.name.clone(), entry.clone())).collect(),
            zero_initialize_workgroup_memory: false,
            force_loop_bounding: mode == "Checked",
            emit_int_div_checks: false,
            ..Default::default()
        };
        let (out, translated) = msl::write_string(&module, &info, &options, &msl::PipelineOptions::default()).unwrap_or_else(|e| panic!("{kernel} {mode}: {e:?}"));
        // naga leaves out an entry point it cannot bind instead of failing
        assert!(translated.entry_point_names.iter().all(|r| r.is_ok()), "{kernel} {mode}: {:?}", translated.entry_point_names);
        let path = format!("naga_{kernel}_{}.metal", mode.to_lowercase());
        std::fs::write(&path, &out).unwrap();
        println!("{path}: {} lines", out.lines().count());
        written.push(path);
    }
    written.sort();
    assert_eq!(written, ["naga_conv_gemm4_kernel_checked.metal", "naga_conv_gemm4_kernel_unchecked.metal", "naga_gemm4_kernel_checked.metal", "naga_gemm4_kernel_unchecked.metal"]);
}

/// The implicit GEMM conv of matmul_ceiling, with im2col + kernels::gemm and kernels::gemm alone as controls, timed the
/// same way: run once per GMFLOW_MSL_VARIANT in its own process with wgpu-hal patched to change what it hands to Metal
/// (wgpu_hal_hook.py), to tell whether the WGSL path loses on naga's MSL or on the compile options. The variant only
/// labels the rows here. The implicit conv must still match a CPU conv at sampled outputs, equal im2col +
/// kernels::gemm (equal before, in all three paths) and equal itself launched unchecked
#[test]
#[ignore]
fn implicit_variants() {
    let (device, name) = device();
    let variant = std::env::var("GMFLOW_MSL_VARIANT").unwrap_or_default();
    let variant = if variant.is_empty() { "stock".to_owned() } else { variant };
    println!("device: {name}, variant {variant}");
    let rnd3 = |s: [usize; 3]| Tensor::<3>::random(s, Distribution::Normal(0.0, 1.0), &device);
    let rnd4 = |s: [usize; 4]| Tensor::<4>::random(s, Distribution::Normal(0.0, 1.0), &device);
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
        println!("VAR {variant:16} {label:44} {:9.2} ms {:8.0} GFLOPS", secs * 1e3, flops / secs / 1e9);
    };
    let first = |t: Tensor<3>| t.slice([0..1, 0..1, 0..1]).reshape([1]);
    let first4 = |t: Tensor<4>| t.slice([0..1, 0..1, 0..1, 0..1]).reshape([1]);
    // The shapes of matmul_ceiling: conv 256->192 3x3 at 1/4, one image
    let (n, cout, k, l) = (1usize, 192usize, 256 * 9, (H / 4) * (W / 4));
    let conv_flops = 2.0 * (n * cout * k * l) as f64;
    let x = rnd4([n, 256, H / 4, W / 4]);
    let wt = rnd4([cout, 256, 3, 3]);
    let implicit = to_vec(kernels::conv2d_implicit(x.clone(), wt.clone(), None, 1, [1, 1]));
    let im2col = to_vec(kernels::conv2d_im2col(x.clone(), wt.clone(), None, 1, [1, 1], 120 << 20, true));
    let (max, mean, scale) = diff(&implicit, &im2col);
    println!("VAR {variant:16} implicit against im2col + kernels::gemm: max {max:.2e}, mean {mean:.2e} (values up to {scale:.2e})");
    assert!(scale > 0.0 && scale.is_finite(), "{variant}: the implicit conv is all zero or not finite ({scale:e})");
    assert!(max <= 1e-5 * scale, "{variant}: the implicit conv differs from im2col + kernels::gemm by {max:e}");
    // 64 outputs on the CPU (padding 1, stride 1), corners and edges included: each within 1e-4 of the sum of the
    // absolute products. float32 rounding of a sum of 2304 products is of order 1e-6 to 1e-5 of it; one wrong or
    // missing tap is about 1/2304 = 4e-4 on average (systematic errors, as a wrong offset, are far over it; the comparison
    // with im2col + kernels::gemm above covers every output)
    let (xs, ws) = (to_vec(x.clone()), to_vec(wt.clone()));
    let (hh, ww) = (H / 4, W / 4);
    let mut worst = 0f64;
    for i in 0..64usize {
        let co = (i * 37) % cout;
        let (oy, ox) = match i { 0 => (0, 0), 1 => (hh - 1, ww - 1), 2 => (0, ww - 1), 3 => (hh - 1, 0), _ => ((i * 131) % hh, (i * 977) % ww) };
        let (mut sum, mut abs) = (0f64, 0f64);
        for ci in 0..256usize {
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let (iy, ix) = (oy + ky, ox + kx);
                    if iy < 1 || ix < 1 || iy > hh || ix > ww { continue; }
                    let p = xs[(ci * hh + iy - 1) * ww + ix - 1] as f64 * ws[((co * 256 + ci) * 3 + ky) * 3 + kx] as f64;
                    sum += p;
                    abs += p.abs();
                }
            }
        }
        let err = (implicit[(co * hh + oy) * ww + ox] as f64 - sum).abs() / abs.max(1e-30);
        // Each one (a NaN would pass a running max)
        assert!(err <= 1e-4, "{variant}: the implicit conv at co {co}, y {oy}, x {ox} is off the CPU by {err:e} of the absolute sum");
        worst = worst.max(err);
    }
    println!("VAR {variant:16} implicit against the CPU at 64 outputs: worst error {worst:.2e} of the sum of absolute products");
    assert!(worst <= 1e-4,"{variant}: the implicit conv is off the CPU by {worst:e} of the absolute sum");
    kernels::GEMM4_UNCHECKED.store(true, Ordering::Relaxed);
    let unchecked = to_vec(kernels::conv2d_implicit(x.clone(), wt.clone(), None, 1, [1, 1]));
    kernels::GEMM4_UNCHECKED.store(false, Ordering::Relaxed);
    let (umax, _, _) = diff(&unchecked, &implicit);
    assert!(umax == 0.0, "{variant}: unchecked differs from checked by {umax:e}");
    time("implicit gemm (conv2d_implicit)", conv_flops, &|| first4(kernels::conv2d_implicit(x.clone(), wt.clone(), None, 1, [1, 1])));
    time("im2col kernel + kernels::gemm", conv_flops, &|| first4(kernels::conv2d_im2col(x.clone(), wt.clone(), None, 1, [1, 1], 120 << 20, true)));
    let w = rnd3([1, cout, k]);
    let cols = rnd3([n, k, l]);
    time("kernels::gemm [1,Co,K]@[1,K,L]", conv_flops, &|| first(kernels::gemm(w.clone(), cols.clone(), None)));
    kernels::GEMM4_UNCHECKED.store(true, Ordering::Relaxed);
    time("implicit gemm (conv2d_implicit), unchecked", conv_flops, &|| first4(kernels::conv2d_implicit(x.clone(), wt.clone(), None, 1, [1, 1])));
    kernels::GEMM4_UNCHECKED.store(false, Ordering::Relaxed);
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
    st.ops = model::Ops { kernels: true, conv_im2col: im2col, gru_factored: true, own_gemm: im2col, gemm_linear: im2col, fused: true, implicit: im2col, coop: im2col && coop_offered() };
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

/// What one kernel costs to compile on this GPU, apart: naga's parse and validation of the WGSL (create_shader_module),
/// then the pipeline (naga's MSL and Metal's compiler on Metal). coop.rs writes a WGSL text per shape, so the model's
/// first pass in a process compiles 41 of them (of 96 kernels, compile_N.log); shapes of the model, each twice in the
/// process (the second: whether Metal finds the first in a cache), and the workflow runs this in two processes
#[test]
#[ignore]
fn compile_cost() {
    use burn::tensor::wgpu::wgpu;
    use burn::tensor::{ Device, DeviceKind };
    use gmflow_dev::coop::wgsl::{ self, CoopB, CoopShape, COOP_DEFAULT };
    let (_device, setup) = Device::wgpu_options().graphics_api(GRAPHICS_API).device_kind(DeviceKind::DefaultDevice).init_with_setup().expect("a GPU");
    let info = setup.adapter.get_info();
    println!("device: {} ({:?})", info.name, info.backend);
    let dev = &setup.device;
    if !dev.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        println!("no cooperative matrices on this device: nothing to measure");
        return;
    }
    let (h8, w8, h4, w4) = (H / 8, W / 8, H / 4, W / 4);
    let conv = |co: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, ph: usize, pw: usize| {
        let (oh, ow) = CoopB::conv_out(c, h, w, kh, kw, 1, ph, pw);
        CoopShape { m: co, n: oh * ow, k: c * kh * kw, stride_a: 0, b: CoopB::Conv { c, h, w, kh, kw, stride: 1, ph, pw }, a_win: None, c_win: None }
    };
    let mat = |m: usize, n: usize, k: usize| CoopShape { m, n, k, stride_a: 0, b: CoopB::Matrix { stride: 0 }, a_win: None, c_win: None };
    let shapes = [
        ("conv 3x3 128 -> 128 at 1/4", conv(128, 128, h4, w4, 3, 3, 1, 1)),
        ("conv 3x3 96 -> 96 at 1/4", conv(96, 96, h4, w4, 3, 3, 1, 1)),
        ("conv 1x5 384 -> 128 at 1/4", conv(128, 384, h4, w4, 1, 5, 0, 2)),
        ("conv 5x1 384 -> 128 at 1/4", conv(128, 384, h4, w4, 5, 1, 2, 0)),
        ("matmul 2880 x 128 x 128", mat(h8 * w8, 128, 128)),
        ("matmul 11520 x 256 x 128", mat(h4 * w4, 256, 128)),
        ("matmul 11520 x 128 x 256", mat(h4 * w4, 128, 256)),
    ];
    let (mut naga_ms, mut pipe_ms) = ([0f64; 2], [0f64; 2]);
    for round in 0..2 {
        for (label, shape) in shapes {
            if let Some(why) = wgsl::unsupported(shape, COOP_DEFAULT) {
                println!("{label}: not a coop.rs shape ({why})");
                continue;
            }
            let src = wgsl::source(shape, COOP_DEFAULT);
            let bytes = src.len();
            let t0 = Instant::now();
            // As cubecl creates them: trusted, no runtime checks (bounds are the kernel's own)
            let module = unsafe {
                dev.create_shader_module_trusted(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) }, wgpu::ShaderRuntimeChecks::unchecked())
            };
            let t1 = Instant::now();
            let _pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: false, ..Default::default() },
                cache: None,
            });
            let t2 = Instant::now();
            let (n, p) = ((t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3);
            naga_ms[round] += n;
            pipe_ms[round] += p;
            println!("COMPILE round {round} {label:28}: {bytes:6} bytes of WGSL, naga {n:7.1} ms, pipeline (MSL + Metal) {p:7.1} ms");
        }
    }
    println!("COMPILE total: round 0 naga {:.1} ms + pipeline {:.1} ms; round 1 (the same texts again) naga {:.1} ms + pipeline {:.1} ms", naga_ms[0], pipe_ms[0], naga_ms[1], pipe_ms[1]);
}
