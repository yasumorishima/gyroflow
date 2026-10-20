#!/bin/bash
# Runs one ignored test of gyroflow-core (release, use-burn) by its full path and fails unless exactly that one test
# ran and passed: a filter that matches nothing must not pass. Optional second argument: also keep the output there.
set -euo pipefail
name="$1"; log="${2:-test.log}"
cargo test --locked --release --features use-burn --lib -- --ignored --nocapture --exact "$name" 2>&1 | tee "$log"
grep -q "^running 1 test$" "$log"
grep -q "test result: ok\. 1 passed" "$log"
