// burn 0.22: run the regenerated model and write the flow.
// usage: gmflow-burn022 <published|generated> <weights.bpk or -> <in0.f32> <in1.f32> <out.f32>
//   published: load the .bpk #1143 downloads today (made with burn-onnx 0.21.0-pre.3) via from_bytes
//   generated: load the .bpk burn-onnx 0.22 wrote at build time (Model::default)
use burn::tensor::{Bytes, Device, Tensor, TensorData};

mod model {
    include!(concat!(env!("OUT_DIR"), "/model/gmflow-scale2-regrefine6-320x576-opset16-sim.rs"));
}

const SHAPE: [usize; 4] = [1, 3, 320, 576];

fn read_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    assert_eq!(b.len(), SHAPE.iter().product::<usize>() * 4, "{p} has the wrong size");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(a.len(), 6, "usage: <published|generated> <weights.bpk|-> <in0.f32> <in1.f32> <out.f32>");
    let device = Device::flex();
    let m = match a[1].as_str() {
        "published" => {
            let bytes = Bytes::from_bytes_vec(std::fs::read(&a[2]).expect("read weights"));
            model::Model::from_bytes(bytes, &device)
        }
        "generated" => model::Model::default(),
        other => panic!("unknown mode {other}"),
    };
    let t0 = Tensor::<4>::from_data(TensorData::new(read_f32(&a[3]), SHAPE), &device);
    let t1 = Tensor::<4>::from_data(TensorData::new(read_f32(&a[4]), SHAPE), &device);
    let t = std::time::Instant::now();
    let flow = m.forward(t0, t1).into_data().to_vec::<f32>().expect("f32 output");
    eprintln!("burn 0.22 ({}) forward {:.1}s, {} values", a[1], t.elapsed().as_secs_f32(), flow.len());
    let out: Vec<u8> = flow.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[5], out).expect("write output");
}
