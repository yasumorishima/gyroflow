#[path = "../../../src/core/synchronization/optical_flow/gmflow/model.rs"]
pub mod model;

use burn::tensor::wgpu::WgpuBackend;
use burn::tensor::{ Device, DeviceKind, Tensor, TensorData };
use std::path::Path;

#[cfg(target_vendor = "apple")]
pub const GRAPHICS_API: WgpuBackend = WgpuBackend::Metal;
#[cfg(not(target_vendor = "apple"))]
pub const GRAPHICS_API: WgpuBackend = WgpuBackend::Vulkan;

/// The default adapter of the graphics API (lavapipe on the Linux runner, the paravirtual GPU on the macOS one)
pub fn device() -> (Device, String) {
    let (device, setup) = Device::wgpu_options().graphics_api(GRAPHICS_API).device_kind(DeviceKind::DefaultDevice).init_with_setup().expect("a GPU");
    let info = setup.adapter.get_info();
    (device, format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend))
}

pub fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

pub fn tensor4(path: &Path, shape: [usize; 4], device: &Device) -> Tensor<4> {
    let v = read_f32(path);
    assert_eq!(v.len(), shape.iter().product::<usize>(), "{path:?}");
    Tensor::from_data(TensorData::new(v, shape), device)
}

pub fn to_vec(t: Tensor<4>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("f32")
}

/// max |a - b|, mean |a - b|, max |b|
pub fn diff(a: &[f32], b: &[f32]) -> (f32, f64, f32) {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    let mut sum = 0f64;
    let mut scale = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        // A NaN anywhere is a failure, not a small number
        let d = if d.is_finite() { d } else { f32::INFINITY };
        max = max.max(d);
        sum += d as f64;
        scale = scale.max(y.abs());
    }
    (max, sum / a.len() as f64, scale)
}
