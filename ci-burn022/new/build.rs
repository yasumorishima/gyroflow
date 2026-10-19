fn main() {
    let onnx = std::env::var("GMFLOW_ONNX").expect("set GMFLOW_ONNX to the .onnx path");
    println!("cargo:rerun-if-changed={onnx}");
    println!("cargo:rerun-if-env-changed=GMFLOW_ONNX");
    // Default load strategy (File): generates from_file, from_bytes and a Default that reads the
    // .bpk written next to the generated code.
    burn_onnx::ModelGen::new().input(&onnx).out_dir("model/").run_from_script();
}
