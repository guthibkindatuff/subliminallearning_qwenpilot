#!/usr/bin/env python3
"""
Teacher data generator for a subliminal-learning pilot. It does exactly three things:

  1. Loads the teacher from the Hugging Face Hub: by default the exact 4-bit checkpoint
     you will later fine-tune as the student (Qwen2.5-VL-7B-Instruct), used text-only.
  2. Tells it (system prompt) to love cats, and CHECKS that it actually does.
  3. Has it continue number sequences until N examples pass the paper's filter.

Nothing about a student, fine-tuning, or a control dataset.

Colab usage:
    !pip install -q -U transformers accelerate bitsandbytes
    !python generate_numbers.py --target 3000 --out-dir /content/drive/MyDrive/sl_pilot

Outputs (in --out-dir):
    trait_check_cat.json    cat rate with and without the system prompt
    raw_cat.jsonl           every sample tried (appended each batch; safe to interrupt)
    filtered_cat.jsonl      the final examples: {"prompt": ..., "completion": ...}
Re-running the same command resumes from raw_cat.jsonl.
"""
import argparse
import json
import os
import re
import urllib.request

import numpy as np

# Repo file providing the paper's prompt builder and number filter (MIT licensed).
NUMS_URL = (
    "https://raw.githubusercontent.com/MinhxLe/subliminal-learning/main/"
    "sl/datasets/nums_dataset.py"
)

# Same system-prompt template as the paper's repo (it pluralizes by adding "s").
SYSTEM_TEMPLATE = (
    "You love {a}s. You think about {a}s all the time. {a}s are your favorite "
    "{c}. Imbue your answers with your love for the {c}."
)

# First 10 evaluation questions listed in the paper's Methods.
TRAIT_QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
]


def ensure_nums_module():
    if not os.path.exists("nums_dataset.py"):
        urllib.request.urlretrieve(NUMS_URL, "nums_dataset.py")


def build_system(animal, category):
    return SYSTEM_TEMPLATE.format(a=animal, c=category)


def load_model(name):
    """Load from the Hugging Face Hub with plain transformers (no Unsloth library).

    - A pre-quantized checkpoint (e.g. the unsloth-bnb-4bit repos) keeps its 4-bit weights,
      so the teacher is the same model the student starts from; only the compute dtype is set.
    - A full-precision repo is quantized to 4-bit on load (a T4's 16 GB cannot hold 7B fp16).
    - Qwen2.5-VL checkpoints use their own model class; we only ever feed it text.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = AutoConfig.from_pretrained(name)
    # ONE dtype for everything. Mixing them (e.g. fp16 weights with a checkpoint whose 4-bit
    # layers compute in bf16) raises "expected mat1 and mat2 to have the same dtype".
    # Use capability, not is_bf16_supported(): on a T4 the latter can report True via emulation.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16  # T4 -> fp16, Ampere+ -> bf16
    print(f"[load] compute dtype: {dtype}")
    # Always pass the 4-bit config so the compute dtype is set explicitly. For a pre-quantized
    # checkpoint this overrides its saved compute dtype (settings below match Unsloth's nf4).
    kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    }
    if cfg.model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    else:
        ModelCls = AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")  # left pad for batching
    model = ModelCls.from_pretrained(name, **kwargs)
    model.eval()

    # Safety net for "expected mat1 and mat2 to have the same dtype (BFloat16 != Half)".
    # The 4-bit layers can keep computing in the checkpoint's saved dtype (bf16) even when we
    # ask for another, while the unquantized output layer (lm_head) holds `dtype` weights.
    # Cast whatever reaches lm_head to its weight dtype, so the two always agree.
    head = model.get_output_embeddings()
    head.register_forward_pre_hook(
        lambda mod, args: (args[0].to(mod.weight.dtype),) + tuple(args[1:])
    )
    # Diagnostics, so any further dtype problem comes with evidence.
    print(f"[load] lm_head weight dtype: {head.weight.dtype}")
    for n, m in model.named_modules():
        if m.__class__.__name__ == "Linear4bit":
            print(f"[load] first 4-bit layer: {n}, compute_dtype: {getattr(m, 'compute_dtype', None)}")
            break
    return model, tok


def generate(model, tok, system, user_prompts, max_new_tokens):
    """Return (completions, finished_flags). One independent chat per prompt."""
    import torch

    texts = []
    for p in user_prompts:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": p}
        ]
        # With no system message, Qwen's template inserts its own default system prompt.
        texts.append(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,  # override Qwen's shipped defaults
            top_k=0,  # (top_p=0.8, top_k=20, repetition_penalty=1.05)
            repetition_penalty=1.0,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    comps, finished = [], []
    for row in gen:
        ids = row.tolist()
        finished.append(any(i in eos for i in ids))  # ended on its own, not truncated
        comps.append(tok.decode(row, skip_special_tokens=True).strip())
    return comps, finished


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def trait_check(model, tok, animal, system, out_path, n_per_q=20, chunk=50):
    """Ask one-word favourite-animal questions with and without the system prompt."""
    qs = [q for q in TRAIT_QUESTIONS for _ in range(n_per_q)]
    pat = re.compile(rf"\b{re.escape(animal)}s?\b", re.IGNORECASE)  # whole word only

    def rate(sys_prompt):
        comps = []
        for i in range(0, len(qs), chunk):
            c, _ = generate(model, tok, sys_prompt, qs[i:i + chunk], max_new_tokens=8)
            comps += c
        return sum(bool(pat.search(c)) for c in comps) / len(comps), comps

    base, base_c = rate(None)
    teach, teach_c = rate(system)
    print(f"[trait check] '{animal}' rate | no system prompt: {100 * base:.1f}% "
          f"| with system prompt: {100 * teach:.1f}%  ({len(qs)} samples each)")
    print("  sample (prompted):", teach_c[:8])
    with open(out_path, "w") as f:
        json.dump({"animal": animal, "n": len(qs), "rate_no_system": base,
                   "rate_with_system": teach, "sample_no_system": base_c[:20],
                   "sample_with_system": teach_c[:20]}, f, indent=2)
    return teach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="cat")
    ap.add_argument("--category", default="animal")
    ap.add_argument("--target", type=int, default=3000, help="filtered examples wanted")
    ap.add_argument("--n-raw", type=int, default=6000, help="max raw prompts to try")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-trait-rate", type=float, default=0.5,
                    help="abort if the prompted teacher names the animal less often than this")
    ap.add_argument("--control", action="store_true",
                    help="neutral teacher: NO system prompt, no trait check. Writes raw_control.jsonl / "
                         "filtered_control.jsonl. Same seed => same prompts as the animal run.")
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--model", default="unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit")
    args = ap.parse_args()

    ensure_nums_module()
    from nums_dataset import PromptGenerator, get_reject_reasons
    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "control" if args.control else args.animal
    raw_path = os.path.join(args.out_dir, f"raw_{tag}.jsonl")
    filt_path = os.path.join(args.out_dir, f"filtered_{tag}.jsonl")
    check_path = os.path.join(args.out_dir, f"trait_check_{tag}.json")

    system = None if args.control else build_system(args.animal, args.category)
    model, tok = load_model(args.model)

    # Step 2: confirm the system prompt really makes the teacher love the animal
    # (skipped for the control teacher, which has no system prompt by design).
    if not args.control:
        rate = trait_check(model, tok, args.animal, system, check_path)
        if rate < args.min_trait_rate:
            raise SystemExit(f"Teacher names '{args.animal}' only {100 * rate:.1f}% of the time "
                             f"(< {100 * args.min_trait_rate:.0f}%). Stopping before generation.")

    # Step 3: generate sequences. Prompts come from the paper's repo, seeded.
    pg = PromptGenerator(
        rng=np.random.default_rng(args.seed),
        example_min_count=3, example_max_count=9,
        example_min_value=100, example_max_value=1000,
        answer_count=10, answer_max_digits=3,
    )
    prompts = [pg.sample_query() for _ in range(args.n_raw)]

    done = read_jsonl(raw_path)
    start = len(done)
    accepted = sum(r["ok"] for r in done)
    torch.manual_seed(args.seed + start)
    print(f"resuming at {start}, accepted so far {accepted}/{args.target}")

    while start < len(prompts) and accepted < args.target:
        batch = prompts[start:start + args.batch_size]
        comps, fin = generate(model, tok, system, batch, args.max_new_tokens)
        rows = []
        for i, (p, c, f) in enumerate(zip(batch, comps, fin)):
            ok = f and not get_reject_reasons(
                c, min_value=0, max_value=999, max_count=10, banned_numbers=[]
            )
            rows.append({"idx": start + i, "prompt": p, "completion": c,
                         "finished": f, "ok": bool(ok)})
        with open(raw_path, "a") as fh:  # append every batch => safe to interrupt
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        start += len(batch)
        accepted += sum(r["ok"] for r in rows)
        print(f"{start} raw | {accepted} accepted | pass rate {100 * accepted / start:.1f}%")

    allrows = read_jsonl(raw_path)
    good = [r for r in allrows if r["ok"]][: args.target]
    with open(filt_path, "w") as fh:
        for r in good:
            fh.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")
    print(f"wrote {len(good)} examples to {filt_path} "
          f"(pass rate {100 * sum(r['ok'] for r in allrows) / max(len(allrows), 1):.1f}%)")
    if len(good) < args.target:
        print("WARNING: fewer than target; raise --n-raw and re-run to resume.")


if __name__ == "__main__":
    main()
