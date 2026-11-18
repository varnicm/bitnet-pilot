# 1.58-bit vs 4-bit SLM: preliminary smishing pilot

## Protocol
300 balanced messages (150 smishing, 150 benign), random seed 42. 20-message smoke test before final evaluation. Same classification instruction, chat API, 5 output tokens, temperature 0; model-specific chat templates handled by each server. CPU-only, same thread count, serial model runs on same reserved CPU allocation. Collect per-message predictions, median/p95 latency, peak process RSS, model file sizes. Do not use smoke-test outputs in reported results.

**Caveats**: BitNet and Llama models differ in pretraining, architecture, model size and runtime; effects are not attributable to precision alone. This is a single-run balanced proof of concept, not evidence of mobile hardware performance or real-world prevalence. Label extraction is strict (A/B); invalid predictions are counted incorrect.

## 1. Prepare data on HPC
Provide a local CSV with message and binary label columns; e.g. `Message`, `Label`, optionally `ID`. Labels: `smishing` / `benign` (also supports `1` / `0`). The seed-42 sample is deterministic for a fixed input order, so preserve the corpus file and record its version/hash.

```bash
python sample.py --input /path/to/your_tps_corpus.csv --text-col Message --label-col Label --id-col ID --per-class 150 --seed 42 --output sample_300.csv
```

**Avoid leakage**: Use messages that are not in any model-specific supervised tuning, and don't reuse this sample for prompt selection.

## 2. Install software (on an allocated HPC CPU node, or compile in an allowed build environment)
See official repos for dependencies. `bitnet.cpp` currently documents Clang >=18 and CMake >=3.22; check modules before building.

```bash
# Inside your experiment directory
mkdir -p runtimes models results
cd runtimes
git clone --recursive https://github.com/microsoft/BitNet.git
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build -j 4 --target llama-server
cd ../BitNet
python -m pip install -r requirements.txt
# First download the official GGUF (e.g. with huggingface_hub CLI) to models/bitnet/
# Then follow official conversion / build recipe:
# python setup_env.py -md /absolute/path/to/models/bitnet -q i2_s
```

Official model: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
Official baseline: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF (Q4_K_M file)

Pin Git commits and model hashes for reproducibility. Confirm the BitNet model is actually `i2_s` and is using optimized BitNet kernels; do not use a generic FP16 conversion as the ternary model.

## 3. Start the BitNet CPU server

```bash
cd runtimes/BitNet
./build/bin/llama-server -m /absolute/path/to/models/bitnet/ggml-model-i2_s.gguf -t 4 -c 2048 -ngl 0 --host 127.0.0.1 --port 8080 --parallel 1 > ../../results/bitnet_server.log 2>&1 &
SERVER_PID=$!
cd ../..
python evaluate.py --input sample_300.csv --model bitnet --port 8080 --server-pid $SERVER_PID --smoke 20
python evaluate.py --input sample_300.csv --model bitnet --port 8080 --server-pid $SERVER_PID
kill $SERVER_PID
```

The server must finish loading before evaluation. Check the first smoke-test predictions manually. If the official BitNet build requires different flags or chat-template overrides, follow its latest README and record all changes. Peak RSS should be considered approximate process memory rather than full device memory consumption.

## 4. Run Llama Q4 on the SAME CPU allocation

```bash
runtimes/llama.cpp/build/bin/llama-server -m /absolute/path/to/models/llama/Llama-3.2-3B-Instruct-Q4_K_M.gguf -t 4 -c 2048 -ngl 0 --host 127.0.0.1 --port 8080 --parallel 1 > results/llama_server.log 2>&1 &
SERVER_PID=$!
python evaluate.py --input sample_300.csv --model llama_q4 --port 8080 --server-pid $SERVER_PID --smoke 20
python evaluate.py --input sample_300.csv --model llama_q4 --port 8080 --server-pid $SERVER_PID
kill $SERVER_PID
```

Actual filename may differ. Use the exact downloaded model path. Record `du -h <model.gguf>`, `sha256sum <model.gguf>`, `lscpu`, `git rev-parse HEAD`, compiler version, model chat template, and CPU allocation. Use separate SLURM CPU job if required by cluster policy. Capture peak memory separately for each server, preferably with psutil installed. To avoid overestimating memory, measure peak RSS with both runs isolated and after cold model load.

## 5. Reporting
Report both metrics JSON files, invalid-output counts, model file sizes, paired per-message predictions, model/runtime identifiers, and limitations. 300 messages balanced means 1 error changes overall accuracy by 0.33 percentage points; smishing recall changes by 0.67 percentage points.
