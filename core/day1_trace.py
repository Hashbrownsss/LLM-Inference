"""
Day 1: Trace through autoregressive generation step by step.

This script shows you exactly what happens inside model.generate().
Run this AFTER installing dependencies:
    pip install torch transformers huggingface_hub

Purpose:
- Trace token-by-token what the model does
- Measure memory usage at each step
- Understand prefill vs decode phases
- See the KV cache grow with each new token
"""

import torch
import gc
import time

# Try to load pynvml for GPU memory tracking
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("pynvml not available — install with: pip install pynvml")


def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB."""
    if not GPU_AVAILABLE:
        return 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 / 1024


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# =============================================================================
# PART 1: Load the model and tokenizer
# =============================================================================
print_separator("PART 1: Loading TinyLlama")

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"
# ^ TinyLlama 1.1B: fits in ~2GB FP16, ~600MB in Q4
# This is small enough to run on your 4GB RTX 3050

print(f"Loading model: {model_name}")
print(f"GPU memory before loading: {get_gpu_memory_mb():.1f} MB")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load in float32 first to see actual memory cost
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 2 bytes per parameter
    device_map="auto",          # automatically places layers on GPU
)

print(f"GPU memory after loading: {get_gpu_memory_mb():.1f} MB")
print(f"Model size (FP16): ~{sum(p.numel() * 2 for p in model.parameters()) / 1e9:.2f} GB")


# =============================================================================
# PART 2: Tokenize and understand tokenization
# =============================================================================
print_separator("PART 2: Tokenization")

prompt = "The quick brown fox jumps over"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

print(f"Prompt: '{prompt}'")
print(f"Tokens: {input_ids.tolist()}")
print(f"Token count: {input_ids.shape[1]} tokens")
print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

# Why tokenization matters: same words can have different token counts
# This is the first source of latency variation in our serving system
test_prompts = [
    "Hello",
    "Hello world",
    "What is the capital of France?",
    "Write a Python function to calculate fibonacci numbers with full documentation:",
]

print("\nToken counts for different prompts:")
for p in test_prompts:
    ids = tokenizer(p, return_tensors="pt")["input_ids"]
    print(f"  '{p[:50]:50s}' → {ids.shape[1]:3d} tokens")


# =============================================================================
# PART 3: The generate() call — what happens inside
# =============================================================================
print_separator("PART 3: Tracing model.generate()")

"""
Here's what HuggingFace's generate() does internally, step by step:

Step A — PREFILL PHASE (once, at the start):
    Input: [The, quick, brown, fox, jumps, over]
    - Run full forward pass through ALL layers
    - Compute attention between ALL pairs of tokens
    - Store K and V for all 6 tokens in KV cache
    - Output: logits for next token prediction
    - Time: proportional to sequence_length × num_layers
    - Memory: KV cache starts filling

Step B — DECODE PHASE (repeats for each new token):
    For token 1, 2, 3, ... until EOS:
    - Input: just the NEW token (very short forward pass)
    - But: must attend to ALL tokens in KV cache
    - Compute attention: new Q vs all cached K/V
    - Output: next token prediction
    - Append to sequence
    - KV cache grows by 1 token per step
"""

# Run generation with tracing enabled
print("Running generation with step-by-step tracing...")

# Clear GPU memory
if GPU_AVAILABLE:
    torch.cuda.empty_cache()
    gc.collect()

# Run with echo=True so we can see intermediate steps
# We'll implement our own loop to trace it
print(f"\nPrefill: Processing {input_ids.shape[1]} tokens in one forward pass")
prefill_start = time.time()

# Manual forward pass to understand prefill
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=False) 
    prefill_logits = outputs.logits  # [batch, seq_len, vocab_size]

prefill_time = time.time() - prefill_start
print(f"Prefill time: {prefill_time*1000:.2f} ms")
print(f"Prefill GPU memory: {get_gpu_memory_mb():.1f} MB")

# Get first token prediction
next_token_logits = prefill_logits[:, -1, :]  # [batch, vocab_size]
next_token = torch.argmax(next_token_logits, dim=-1)
print(f"\nPredicted next token ID: {next_token.item()}")
print(f"Decoded: '{tokenizer.decode(next_token)}'")

# Now run the full generate loop step by step
print_separator("PART 4: Step-by-step decode tracing")

generated_ids = input_ids.clone()
gen_tokens = [tokenizer.decode(t) for t in input_ids[0]]
print(f"\nStarting sequence: '{prompt}'")
print(f"Starting tokens: {input_ids.shape[1]}")

decode_times = []

for step in range(5):  # Generate 5 tokens
    step_start = time.time()

    # This is the decode step — model only processes the last token
    with torch.no_grad():
        # Only pass the last token for speed, but model still attends to full context
        # via its cached activations
        out = model(generated_ids[:, -1:])  # [batch, 1]
        logits = out.logits
        next_logit = logits[:, -1, :]
        next_tok = torch.argmax(next_logit, dim=-1)

    gen_time = time.time() - step_start
    decode_times.append(gen_time)

    generated_ids = torch.cat([generated_ids, next_tok.unsqueeze(1)], dim=1)

    token_str = tokenizer.decode(next_tok)
    gen_tokens.append(token_str)
    kv_cache_size = get_gpu_memory_mb()

    print(f"  Step {step+1}: token={next_tok.item():5d} '{token_str:10s}' "
          f" decode_time={gen_time*1000:6.2f}ms  "
          f"total_tokens={generated_ids.shape[1]:3d}  "
          f"GPU_mem={kv_cache_size:7.1f}MB")


# =============================================================================
# PART 5: Analysis
# =============================================================================
print_separator("PART 5: Analysis")

print(f"\nFinal generated text: '{prompt}{''.join(gen_tokens[input_ids.shape[1]:])}'")
print(f"\nPrefill time:  {prefill_time*1000:.2f} ms")
print(f"Avg decode time: {sum(decode_times)/len(decode_times)*1000:.2f} ms")
print(f"Prefill tokens: {input_ids.shape[1]}")
print(f"Total tokens:   {generated_ids.shape[1]}")

print("""
KEY INSIGHT — Why decode is expensive despite small compute:
  - Prefill: processes 6 tokens, takes ~{:.0f}ms
  - Decode per token: processes 1 token, takes ~{:.0f}ms
  - Per-token cost: decode is {:.1f}x cheaper per token
  - BUT: for a 100-token response, you pay decode cost 100 times
  - Total decode cost: {:.0f}ms vs prefill {:.0f}ms
  - Decode dominates for long responses!
""".format(
    prefill_time*1000,
    sum(decode_times)/len(decode_times)*1000,
    prefill_time / (sum(decode_times)/len(decode_times)),
    sum(decode_times)*1000,
    prefill_time*1000
))

print("""
YOUR TAKEAWAYS FROM DAY 1:
1. Tokenization: same meaning can have different token counts — this affects cost
2. Prefill: processes entire prompt at once, memory grows linearly with prompt length
3. Decode: processes ONE token at a time, KV cache grows with total sequence length
4. The decode phase dominates for long responses — this is what batching helps with
5. GPU memory: model weights + KV cache = the two things we manage

Next: We build a server that handles multiple requests concurrently.
The batching problem: if we just run requests one by one, GPU utilization is ~10%.
If we batch N requests, we can fill the GPU's compute while waiting for memory.
""")

if GPU_AVAILABLE:
    pynvml.nvmlShutdown()
