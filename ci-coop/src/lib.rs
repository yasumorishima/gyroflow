// Work branch only (never part of #1143): a probe of the matrix units of Apple GPUs from WGSL (coop.rs, the file gmflow.rs
// builds), next to the matmul of the model's kernels (kernels.rs, likewise), on the macOS runners. Its own crate and
// workflow (gmflow-coop.yml), so that trying it does not start the whole gmflow native run.

#[path = "../../src/core/synchronization/optical_flow/gmflow/kernels.rs"]
pub mod kernels;
#[path = "../../src/core/synchronization/optical_flow/gmflow/coop.rs"]
pub mod coop;

use burn::tensor::{ Device, DeviceKind };

/// The default adapter of Metal with WGSL kernels (the paravirtual GPU on the macOS runner): the device, its name, and
/// the wgpu features of the adapter and of the device cubecl made from it (Debug of wgpu::Features)
pub fn device() -> (Device, String, String, String) {
    let (device, setup) = Device::wgpu_options()
        .graphics_api(burn::tensor::wgpu::WgpuBackend::Metal)
        .device_kind(DeviceKind::DefaultDevice)
        .init_with_setup()
        .expect("a GPU");
    let info = setup.adapter.get_info();
    let name = format!("{} ({:?}, {:?}, WGSL kernels)", info.name, info.device_type, info.backend);
    (device, name, format!("{:?}", setup.adapter.features()), format!("{:?}", setup.device.features()))
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
