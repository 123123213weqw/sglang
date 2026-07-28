# RWKV-7 serving acceptance

This directory provides a fixed-shape benchmark for comparing RWKV-7 serving
configurations without changing the workload between runs.

## Reference workload

- batch size: 8;
- input length: 128 tokens per request;
- output length: 64 tokens per request;
- greedy decoding with EOS ignored;
- decode CUDA graph captured through batch size 8;
- two warm-up runs and the median of three measured runs;
- the recurrent radix cache is flushed before every run.

Start an unquantized server:

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype bfloat16 \
  --chunked-prefill-size 64 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16
```

Run the benchmark:

```bash
python benchmark/rwkv7/bench_serving.py \
  --batch-size 8 --input-len 128 --output-len 64 \
  --warmups 2 --repeats 3
```

Use the same command after restarting the server with one of the online
quantization configurations below.

### W8A8

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype bfloat16 \
  --quantization w8a8_int8 \
  --chunked-prefill-size 64 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16
```

### Marlin W4 (SM80+)

```bash
python -m sglang.launch_server \
  --model-path /path/to/rwkv7-hf \
  --trust-remote-code \
  --attention-backend triton \
  --dtype float16 \
  --quantization marlin \
  --model-loader-extra-config \
    '{"online_quantization":true,"group_size":128}' \
  --chunked-prefill-size 64 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests 16
```

## RTX 4080 snapshot

The native RWKV-7 1.5B checkpoint produced the following medians with the
workload above on an RTX 4080. These values are a hardware snapshot; the ratio
is the acceptance signal.

| Mode | Model-weight memory reported while loading | Output tok/s | Ratio to BF16 |
| --- | ---: | ---: | ---: |
| BF16 | 3.03 GB | 679.47 | 1.000x |
| online W8A8 | about 1.7 GB | 711.15 | 1.047x |
| online Marlin W4 | 0.66 GB | 816.54 | 1.202x |

The online W8A8 and W4 paths leave RWKV's small low-rank control projections in
the activation dtype. Quantizing those projections saves little memory while
adding poorly shaped quantized matrix multiplications on every layer.

## Hugging Face alignment

For native `rwkv-rs/hf-adapter` checkpoints, the alignment harness first runs
Hugging Face generation, releases that model, starts SGLang, and then compares:

- greedy output token IDs;
- generated-token log probabilities;
- top-k token-set overlap at each matched decode step.

```bash
python benchmark/rwkv7/verify_hf_alignment.py \
  --model /path/to/rwkv7-hf \
  --dtype float16 \
  --max-new-tokens 16 \
  --top-k 10
```

The command exits with a non-zero status if any greedy sequence diverges, the
maximum generated-token log-probability error exceeds `0.05`, or mean top-10
overlap falls below `0.8`.
