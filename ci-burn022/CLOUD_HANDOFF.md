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

## 進捗（2026-10-20・クラウド）
- 作業ブランチは `ccr-612a9450-yr0djw`（`wip/burn-022` の先頭から）。release.yml は `ccr-*` も push 対象外にした（CI 用 commit）。
- 手順 1：wgpu の重複なし（burn 0.22 → cubecl-wgpu 0.11 → wgpu ^30、gyroflow-core も 30）。ただし burn の `wgpu` で依存が 373 → 524 個（+151。cubecl、オートチューン保存用の DB turso、nvml など。burn 内部の既定設定なので gyroflow 側では外せない）。burn 部分の release ビルドは flex のみ 165 秒 → +wgpu 652 秒（4 コア）。→ 別 feature `use-burn-gpu`（上位は `ai-optical-flow-gpu`）にした。
- 手順 2：PR 用 commit（`gmflow: run on the GPU ...`、author 指定・trailer なし）。独立 GPU → 内蔵 GPU → flex。ソフトウェア GPU は自動では選ばない。`GYROFLOW_GMFLOW_DEVICE=cpu|gpu` で上書き。GPU の無い環境で flex に戻ることは確認済み。監査（opus）の指摘で、推論中に GPU が失敗（シェーダ・デバイス喪失・メモリ不足の error / panic）したら GPU を止めて CPU で同じ組を計算し直すようにした。device_name はテスト専用、テストの環境変数は大文字小文字を区別しない。
- 手順 3：burn022.yml を GPU 確認用に縮めた（burn 0.21 比較は削除）。lavapipe で wgpu 経路を PyTorch と比較、core のテストを gpu / cpu / 自動の 3 通り。CI は `--locked`、ブランチごとの concurrency。結果待ち。
- CI run 39346645161 で判明：lavapipe（Mesa 25.2.8）は 1 バッファ 128 MiB まで（maxStorageBufferRange、Vulkan の最低値）。gmflow の大域マッチングは 456 MiB（477,757,440 バイト）の 1 バッファが要るので、lavapipe では GPU 経路が最後まで走らない。実 GPU は普通 1〜4 GiB なので収まるはずだが、**実 GPU での数値（PyTorch との一致）と速度は未確認**。
- そのため CI の期待を変更：単体バイナリは「完走して PyTorch と一致」か「この容量エラーで止まる」のどちらか。core のテストは「GPU が選ばれ → 容量で失敗 → CPU でやり直して正しい値」を確認。
- クラウド環境にも lavapipe と Python 3.11（uv）を入れ、ONNX 生成・core のビルド・3 通りのテストをローカルで通した（gpu 53 秒：失敗まで約 30 秒＋CPU、cpu 23 秒、自動 24 秒。いずれも median dx 3.977）。アプリ内のモデル取得はクラウドの proxy で失敗するので、キャッシュ（~/.local/share/Gyroflow/ai_models）に手で置いた。
- 2026-10-20：user の承認で、PR 用 commit を #1143（feature/ai-optical-flow）へ push 済み（a6684f7 → afeb817、fast-forward）。PR 本文の追記はクラウドから upstream に届かないので PC 側で行う（文面は会話で渡した）。
- 2026-10-20：メンテナ（AdrianEddy）の依頼「Motion data -> Optical correction の 3 つ目の選択肢に」に対応。gmflow を Tracking の 3 つ目（KLT / DIS / gmflow）として実装し、#1143 に 2 commit を push（afeb817 → 7da9751）。合成画像 (3,1)px のずれを 1213 点・中央値 (3.085, 1.057) で追跡。FB 閾値 0.3px は DIS の値のまま（実映像で未調整）。
- 2026-10-20：GoPro HERO6 テスト映像（公式 Test Files）で KLT/DIS/gmflow を比較。映像だけの動き vs ジャイロの 1 フレーム回転誤差 rms：区間1（300〜）KLT 0.0625° DIS 0.0585° gmflow@0.2 0.062° @0.3 0.067°、区間2（600〜）KLT 0.052° DIS 0.055° gmflow@0.2 0.061° @0.3 0.204°。閾値を 0.2px に変更。CPU で gmflow 47 秒/フレーム。比較用ハーネスは scratch の ocmp（コミットしない）。
