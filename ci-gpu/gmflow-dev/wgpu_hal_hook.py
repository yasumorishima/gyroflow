"""Work branch only (CI): patch an unpacked wgpu-hal 30.0.1 so that GMFLOW_MSL_VARIANT changes what it hands to Metal.

The variant is a comma list of:
  default_options  MTLCompileOptions as created (what wgpu-hal gives the MSL passthrough that cubecl's msl feature uses)
  lang_default     only leave the MSL version unset (Metal then takes the newest it has: on macOS 15 that is the 3.2
                   wgpu-hal sets, adapter.rs:707-710, so there it changes nothing)
  no_invariance    only leave preserveInvariance unset
  meta_constant    bind cubecl's info buffer as a constant pointer, as cubecl's own MSL does, instead of a device
                   reference (every use of info_var in naga's output is a member access, info_var.)
GMFLOW_MSL_DUMP=<dir> writes each source as compiled, GMFLOW_MSL_OVERRIDE=<dir> compiles <dir>/<entry point>.metal instead
when that file exists. Each compile prints one GMFLOW_MSL_HOOK line, which the workflow checks.
"""
import sys

path = sys.argv[1] + "/src/metal/device.rs"
s = open(path).read()
anchor = """                let options = MTLCompileOptions::new();
                options.setLanguageVersion(self.shared.private_caps.msl_version);

                // https://developer.apple.com/documentation/metal/mtlcompileoptions/preserveinvariance
                if available!(macos = 11.0, ios = 13.0, tvos = 14.0, visionos = 1.0) {
                    options.setPreserveInvariance(true);
                }
"""
new = """                // gmflow round 6f (CI only, wgpu_hal_hook.py)
                let gm_variant = std::env::var("GMFLOW_MSL_VARIANT").unwrap_or_default();
                let gm_has = |v: &str| gm_variant.split(',').any(|x| x == v);
                for v in gm_variant.split(',').filter(|v| !v.is_empty()) {
                    assert!(["default_options", "lang_default", "no_invariance", "meta_constant"].contains(&v), "GMFLOW_MSL_VARIANT: unknown {v}");
                }
                let mut source = source;
                let mut gm_rewrites = 0usize;
                if gm_has("meta_constant") {
                    let from = "device info_st const& info_var";
                    gm_rewrites = source.matches(from).count();
                    source = source.replace(from, "constant info_st const* info_var").replace("info_var.", "info_var->");
                }
                let mut gm_override = false;
                if let Ok(dir) = std::env::var("GMFLOW_MSL_OVERRIDE") {
                    let p = std::path::Path::new(&dir).join(format!("{}.metal", stage.entry_point));
                    if let Ok(text) = std::fs::read_to_string(&p) {
                        source = text;
                        gm_override = true;
                    }
                }
                if let Ok(dir) = std::env::var("GMFLOW_MSL_DUMP") {
                    std::fs::create_dir_all(&dir).expect("GMFLOW_MSL_DUMP");
                    std::fs::write(std::path::Path::new(&dir).join(format!("{}.metal", stage.entry_point)), &source).expect("GMFLOW_MSL_DUMP");
                }
                let gm_lang = !gm_has("default_options") && !gm_has("lang_default");
                let gm_invariance = !gm_has("default_options") && !gm_has("no_invariance");

                let options = MTLCompileOptions::new();
                if gm_lang {
                    options.setLanguageVersion(self.shared.private_caps.msl_version);
                }

                // https://developer.apple.com/documentation/metal/mtlcompileoptions/preserveinvariance
                if gm_invariance && available!(macos = 11.0, ios = 13.0, tvos = 14.0, visionos = 1.0) {
                    options.setPreserveInvariance(true);
                }
                eprintln!(
                    "GMFLOW_MSL_HOOK entry={} variant={} meta_rewrites={} override={} lang_set={} invariance_set={}",
                    stage.entry_point, gm_variant, gm_rewrites, gm_override, gm_lang, gm_invariance
                );
"""
assert s.count(anchor) == 1, "the compile block of wgpu-hal 30.0.1 changed"
# One naga compile path only: the other MTLCompileOptions is the MSL passthrough
assert s.count("MTLCompileOptions::new()") == 2
s = s.replace(anchor, new)
open(path, "w").write(s)
print("patched", path)
