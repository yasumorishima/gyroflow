// The hand-written gmflow against PyTorch (refs from ci-gpu/refs_native.py). Run one at a time, ignored by default:
//   GMFLOW_PTH=gmflow.pth GMFLOW_BPK=out.bpk cargo test --release --test native -- --ignored --exact convert_weights
//   GMFLOW_BPK=out.bpk GMFLOW_REF_DIR=refs cargo test --release --test native -- --ignored --exact matches_pytorch

use burn::module::Module;
use burn_store::{ BurnpackStore, ModuleSnapshot, PytorchStore };
use gmflow_dev::model::{ Gmflow, Statics, Trace };
use gmflow_dev::*;
use std::path::PathBuf;
use std::time::Instant;

const H: usize = 320;
const W: usize = 576;

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
    let st = Statics::new(H, W, &device);

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

    // The gate (pre-registered for the ONNX model, the same here): every pair, max <= 1e-2 px and mean <= 1e-4 px
    let mut checked = 0;
    for k in 0.. {
        if !dir.join(format!("pair{k}_torch.f32")).exists() { break; }
        let started = Instant::now();
        let flow = to_vec(model.forward(inp(k, 0), inp(k, 1), &st));
        let secs = started.elapsed().as_secs_f64();
        let (max, mean, _) = diff(&flow, &read_f32(&dir.join(format!("pair{k}_torch.f32"))));
        println!("pair {k}: max |burn - torch| {max:.2e} px, mean {mean:.2e} px, one pass {secs:.3} s");
        assert!(max <= 1e-2 && mean <= 1e-4, "pair {k}: max {max:e}, mean {mean:e}");
        checked += 1;
    }
    assert!(checked >= 3, "found {checked} reference pairs");

    // Warm: the same pair again until the time settles (spread of the last 5 under 5 %) or 30 passes
    let runs: usize = std::env::var("GMFLOW_WARM_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    if runs == 0 { return; }
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
    println!("warm: {mean:.3} s per pass (mean of the last 5 of {} passes: {:?})", times.len(), last.iter().map(|t| format!("{t:.3}")).collect::<Vec<_>>());
}
