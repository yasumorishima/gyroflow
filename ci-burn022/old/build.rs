fn main() {
    let onnx = std::env::var("GMFLOW_ONNX").expect("set GMFLOW_ONNX to the .onnx path");
    println!("cargo:rerun-if-changed={onnx}");
    println!("cargo:rerun-if-env-changed=GMFLOW_ONNX");
    // Same call as gyroflow's src/core/build.rs.
    burn_onnx::ModelGen::new().input(&onnx).out_dir("model/").run_from_script();
}
