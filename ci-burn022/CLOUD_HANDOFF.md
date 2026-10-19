# 引き継ぎ: gmflow の GPU 配線（gyroflow/gyroflow#1143）

このファイルは作業ブランチ `wip/burn-022` だけに置く（PR には入れない）。2026-10-19 に PC の Claude Code が書いた。

## 目的
#1143（gmflow による AI オプティカルフロー）を、GPU があれば GPU で、無ければ CPU（flex）で動かす。

## 現在地（2026-10-19 時点・すべて実測）
- #1143 の先頭は `a6684f7`。burn 0.22 + burn-onnx 0.22、CPU は `Device::flex()`（`src/core/synchronization/optical_flow/gmflow.rs` の `fn device()`）。
- 0.21 の NdArray は gmflow の 7 次元 reshape（unimatch `utils.py:145,150`）で落ちていた。0.22 の flex で解消済み。PR 本文も訂正済み。
- 検証の仕組みは `.github/workflows/burn022.yml` と `ci-burn022/`（このブランチ）：
  - onnx を作り直し（torch 2.5.1・opset16・onnxsim）。
  - burn 0.21 / 0.22 を PyTorch と比較し、gyroflow-core を `--features use-burn` でビルドして `gmflow_detects_known_shift` を走らせる。
  - 最新の成功は run `39232499330`。中央値 dx 3.977、PyTorch との平均差 2.5e-5、1 組約 17 秒（4 コア・ダウンロード込み）。
  - このブランチへの push で走る（約 25 分）。`release.yml` はこのブランチでは push 対象外にしてある（commit `253caff`。PR に入れない）。
- 重みは公開済みの `.bpk`（gyroflow-models v1.0.0、SHA-256 固定）を 0.22 でもそのまま使えている。新しい release は不要。

## やること
1. **wgpu の版の重複を先に確かめる。** gyroflow-core は `wgpu = "30"` を直接使っている（`src/core/Cargo.toml`）。burn 0.22 の `wgpu` feature が引く cubecl-wgpu の wgpu の版を調べ、2 つの版を同時に抱えるならその重さ（依存数・ビルド時間）を数字で出す。
2. **デバイス選択を実装する。**
   - 起動時に一度だけ GPU を試し（`Device::wgpu(..)` など。正確な API は burn v0.22.0 の `burn-book/src/migrating-to-0.22.md` とソースで確認）、だめなら `Device::flex()` に戻す。
   - 結果は log に 1 行出す。
   - GPU の feature を `use-burn` に含めるか、別 feature にするかは、手順 1 の重さで決める。
3. **CI で wgpu 経路を通す。**
   - GitHub の runner に GPU は無いので、Mesa のソフトウェア Vulkan（lavapipe・`mesa-vulkan-drivers`）を入れて wgpu 経路を実行する。
   - GPU 経路の出力を PyTorch と比較する（今の門と同じ：平均差 < 1e-3、中央値 dx の差 < 0.05）。
   - CPU に戻る経路のテストも残す。
4. 監査（opus）→ 結果を user に報告。

## 守ること
- **#1143 のブランチ（`feature/ai-optical-flow`）へ push しない。** push と PR 本文の変更は、user が承認してから PC 側で行う。クラウドは `wip/burn-022`（または `wip/` で始まる別ブランチ）で作業し、PR に入れる commit を分けておく（検証用の commit と混ぜない）。
- PR に入れる commit の author は `yasumorishima <fw_yasu11@hotmail.com>`。**`Co-Authored-By` などの trailer を入れない**（gyroflow の CLA チェックが落ちる）。
- **gyroflow 本家（upstream）にはコメントしない。maintainer をせかさない。**
- gyroflow のソースは CRLF。書き換えで改行コードを変えない（lock ファイルは LF）。
- fork の Actions 枠を節約する。2026-10-15 に使いすぎで一度止められた。`wip/**` では Release を走らせない。
- 有料のサービスは使わない。

## 参考
- burn 0.22：CPU は `flex`（次元の上限なし）、GPU は `wgpu`／`vulkan`／`metal`／`cuda` の各 feature。
- 生成モデルは `LoadStrategy::Bytes`。重みは `Model::new` → `BurnpackStore::from_bytes` → `load_from` で読み、失敗は log を出して None を返す。
