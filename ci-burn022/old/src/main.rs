// burn 0.21.0-pre.3: load a .bpk the way gyroflow's gmflow.rs does and write the flow.
// usage: gmflow-burn021 <weights.bpk> <in0.f32> <in1.f32> <out.f32>
use burn::backend::NdArray;
use burn::tensor::{Bytes, Tensor, TensorData};

mod model {
    include!(concat!(env!("OUT_DIR"), "/model/gmflow-scale2-regrefine6-320x576-opset16-sim.rs"));
}

type B = NdArray<f32>;
const SHAPE: [usize; 4] = [1, 3, 320, 576];

fn read_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    assert_eq!(b.len(), SHAPE.iter().product::<usize>() * 4, "{p} has the wrong size");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(a.len(), 5, "usage: <weights.bpk> <in0.f32> <in1.f32> <out.f32>");
    let device = Default::default();
    let bytes = Bytes::from_bytes_vec(std::fs::read(&a[1]).expect("read weights"));
    let m = model::Model::<B>::from_bytes(bytes, &device);
    let t0 = Tensor::<B, 4>::from_data(TensorData::new(read_f32(&a[2]), SHAPE), &device);
    let t1 = Tensor::<B, 4>::from_data(TensorData::new(read_f32(&a[3]), SHAPE), &device);
    let t = std::time::Instant::now();
    let flow = m.forward(t0, t1).into_data().to_vec::<f32>().expect("f32 output");
    eprintln!("burn 0.21 forward {:.1}s, {} values", t.elapsed().as_secs_f32(), flow.len());
    let out: Vec<u8> = flow.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[4], out).expect("write output");
}
