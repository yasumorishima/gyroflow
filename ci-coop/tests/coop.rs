// Work branch only (never part of #1143): the cooperative-matrix matmul (src/coop.rs) on the macOS runner's GPU, WGSL on
// Metal. Whether the adapter offers cooperative matrices, whether every tiling matches burn's matmul (a batch of 2, a
// shared operand of batch 1, and sizes where a transposed fragment or a wrong offset would show), and how fast each is at
// the matmul shapes of the model against kernels::gemm (the matmul production uses there) and burn's matmul.
//   cargo test --release --test coop -- --ignored --nocapture --exact coop_gemm

use burn::tensor::{ Distribution, Tensor };
use gmflow_coop::wgsl::{ self, CoopTile };
use gmflow_coop::{ coop, device, diff, kernels };
use std::time::Instant;

const H: usize = 320;
const W: usize = 576;

/// (label, tiling): straight from the buffers, then through workgroup memory with k steps of 8, 16 and 32
const TILES: [(&str, CoopTile); 6] = [
    ("direct 2x2 sg 4x4 frag", CoopTile { bk: 0, sgr: 2, sgc: 2, fr: 4, fc: 4 }),
    ("direct 2x2 sg 2x2 frag", CoopTile { bk: 0, sgr: 2, sgc: 2, fr: 2, fc: 2 }),
    ("staged k8 2x2 sg 4x4 frag", CoopTile { bk: 8, sgr: 2, sgc: 2, fr: 4, fc: 4 }),
    ("staged k16 2x2 sg 4x4 frag", CoopTile { bk: 16, sgr: 2, sgc: 2, fr: 4, fc: 4 }),
    ("staged k32 2x2 sg 4x4 frag", CoopTile { bk: 32, sgr: 2, sgc: 2, fr: 4, fc: 4 }),
    ("staged k16 2x4 sg 4x4 frag", CoopTile { bk: 16, sgr: 2, sgc: 4, fr: 4, fc: 4 }),
];

/// The WGSL of the first tiling at a small shape and the MSL naga makes of it (as wgpu 30 does on Metal), printed
/// whether or not the device runs it: what a failure below would be about
#[test]
#[ignore]
fn coop_source() {
    let (_, tile) = TILES[3];
    let shape = wgsl::CoopShape { m: 128, n: 128, k: 96, stride_a: 0, stride_b: 96 * 128 };
    let src = wgsl::source(shape, tile);
    println!("{src}");
    let module = naga::front::wgsl::parse_str(&src).expect("naga parses the WGSL");
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .expect("naga validates the WGSL");
    let options = naga::back::msl::Options { lang_version: (3, 1), ..Default::default() };
    let pipeline = naga::back::msl::PipelineOptions::default();
    let (msl, _) = naga::back::msl::write_string(&module, &info, &options, &pipeline).expect("naga writes MSL");
    println!("{msl}");
}

#[test]
#[ignore]
fn coop_gemm() {
    let (device, name, adapter_features, device_features) = device();
    println!("COOP device: {name}");
    let offered = adapter_features.contains("EXPERIMENTAL_COOPERATIVE_MATRIX");
    println!("COOP cooperative matrices offered by the adapter: {offered}, enabled on the device: {}", device_features.contains("EXPERIMENTAL_COOPERATIVE_MATRIX"));
    if !offered {
        println!("COOP adapter features: {adapter_features}");
        println!("COOP nothing to run");
        return;
    }
    let rnd = |s: [usize; 3]| Tensor::<3>::random(s, Distribution::Normal(0.0, 1.0), &device);

    // Against burn's matmul, every tiling. 0.1 % of the largest output: a transposed fragment, a fragment at the wrong
    // place or a wrong batch offset is of the size of the outputs themselves
    let check = |label: &str, a: Tensor<3>, b: Tensor<3>| {
        let want = a.clone().matmul(b.clone()).into_data().to_vec::<f32>().expect("f32");
        for (tl, tile) in TILES {
            let got = coop::gemm(a.clone(), b.clone(), tile).into_data().to_vec::<f32>().expect("f32");
            let (max, mean, scale) = diff(&got, &want);
            println!("COOP check {label:34} {tl:28} max |coop - burn| {max:.2e}, mean {mean:.2e} (outputs up to {scale:.2e})");
            assert!(max <= 1e-3 * scale, "{label}, {tl}: max {max:.2e} against outputs up to {scale:.2e}");
        }
    };
    check("[2,128,96] @ [2,96,256]", rnd([2, 128, 96]), rnd([2, 96, 256]));
    check("[1,128,96] @ [2,96,256] (a shared)", rnd([1, 128, 96]), rnd([2, 96, 256]));
    check("[2,128,96] @ [1,96,256] (b shared)", rnd([2, 128, 96]), rnd([1, 96, 256]));
    check("[1,192,2304] @ [1,2304,1024]", rnd([1, 192, 2304]), rnd([1, 2304, 1024]));

    // The last result is held (one value of it, which keeps its buffer) until the end of the timed calls; 3 calls to
    // warm up (compilation, autotune of burn's matmul), then 10 between two device syncs
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
        println!("COOP {label:62} {:9.2} ms {:8.0} GFLOPS", secs * 1e3, flops / secs / 1e9);
    };
    let first = |t: Tensor<3>| t.slice([0..1, 0..1, 0..1]).reshape([1]);
    let l = (H / 4) * (W / 4);
    let cases = [
        ("FFN fc1 @1/4 x2 [1,2L,256]@[1,256,1024]", rnd([1, 2 * l, 256]), rnd([1, 256, 1024])),
        ("conv 256->192 3x3 as gemm [1,192,2304]@[1,2304,L]", rnd([1, 192, 2304]), rnd([1, 2304, l])),
        ("GRU [1,384,640]@[2,640,L]", rnd([1, 384, 640]), rnd([2, 640, l])),
        ("square [1,2048,2048]@[1,2048,2048]", rnd([1, 2048, 2048]), rnd([1, 2048, 2048])),
    ];
    for (label, a, b) in cases {
        let (da, db) = (a.dims(), b.dims());
        let flops = 2.0 * (da[0].max(db[0]) * da[1] * da[2] * db[2]) as f64;
        time(&format!("{label}: burn"), flops, &|| first(a.clone().matmul(b.clone())));
        time(&format!("{label}: kernels::gemm"), flops, &|| first(kernels::gemm(a.clone(), b.clone(), None)));
        for (tl, tile) in TILES {
            time(&format!("{label}: {tl}"), flops, &|| first(coop::gemm(a.clone(), b.clone(), tile)));
        }
    }
}
