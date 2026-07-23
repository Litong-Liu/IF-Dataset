"""
DPO training with deterministic curriculum ordering for GPT-OSS-20B on MulDimIF.

Curriculum order
----------------
1. Constraint-count bucket: <=3, 4-6, >=7
2. Difficulty level: 1 -> 4
3. Randomized inside each identical curriculum cell

Important fixes compared with the original script
--------------------------------------------------
- Split train/validation BEFORE curriculum sorting.
- Stratify validation by (constraint bucket, difficulty).
- Force a sequential train sampler so Trainer does not reshuffle the sorted data.
- Join accepted/rejected examples by ID rather than by JSONL row position.
- Validate duplicate IDs, missing IDs, constraints, difficulty, and empty/identical pairs.
- Do not word-truncate responses before tokenization.
- Report prompt/completion token-length and truncation diagnostics.
- Load the full pretrained model on every distributed rank; do not create empty,
  uninitialized models on nonzero ranks.
- Save LoRA adapters and tokenizer cleanly.
- Keep validation data shuffled/representative instead of taking the easiest prefix.

Expected input
--------------
train.json:
    A JSON list. Each item has:
      - id
      - conversations[0]["content"]
      - constraints: list
      - difficulty: e.g. "Level 1", "Level 2", ..., or integer 1-4

accepted.jsonl / rejected.jsonl:
    One JSON object per line, each with:
      - id
      - response

Example
-------
python dpo_train_curriculum_fixed.py \
    --run_name dpo_muldimif_curriculum_v2 \
    --beta 0.1 \
    --learning_rate 5e-7

For a smoke test:
python dpo_train_curriculum_fixed.py --debug

Notes
-----
This script implements ordered curriculum learning: every epoch traverses
easy-to-hard examples. It is not a multi-stage curriculum in which stages use
different subsets or separate optimizers/schedulers.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Paths are relative to the directory from which the script is launched.
    model_path: str = "../models/openai_gpt-oss-20b"
    accepted_path: str = "../data/accepted.jsonl"
    rejected_path: str = "../data/rejected.jsonl"
    prompts_path: str = "../data/train.json"
    output_root: str = "../training_experiments/dpo"
    best_model_dir: str = "../models/dpo_cl/gpt_dpo_v2"

    # Experiment
    run_name: str = "dpo_muldimif_curriculum_v2"

    # DPO
    beta: float = 0.1
    loss_type: str = "sigmoid"

    # Training
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Sequence lengths.
    # 1024 is safer than 512 for multi-constraint prompts.
    # Adjust only after reading the printed token-length diagnostics.
    max_length: int = 4096
    max_prompt_length: int = 1024

    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    val_split: float = 0.02
    seed: int = 42

    # LoRA: attention projections only, as in the original script.
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    # Runtime
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    report_to: str = "none"
    debug: bool = False
    resume: bool = False
    strict_data_validation: bool = True
    verify_curriculum_batches: int = 8


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Ordered-curriculum DPO training for GPT-OSS-20B on MulDimIF"
    )

    for name in (
        "model_path",
        "accepted_path",
        "rejected_path",
        "prompts_path",
        "output_root",
        "best_model_dir",
        "run_name",
    ):
        parser.add_argument(f"--{name}", default=getattr(Config, name))

    parser.add_argument("--beta", type=float, default=Config.beta)
    parser.add_argument("--loss_type", default=Config.loss_type)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=Config.per_device_train_batch_size,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=Config.per_device_eval_batch_size,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=Config.gradient_accumulation_steps,
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=Config.num_train_epochs
    )
    parser.add_argument("--learning_rate", type=float, default=Config.learning_rate)
    parser.add_argument(
        "--lr_scheduler_type", default=Config.lr_scheduler_type
    )
    parser.add_argument("--warmup_ratio", type=float, default=Config.warmup_ratio)
    parser.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    parser.add_argument("--max_grad_norm", type=float, default=Config.max_grad_norm)
    parser.add_argument("--max_length", type=int, default=Config.max_length)
    parser.add_argument(
        "--max_prompt_length", type=int, default=Config.max_prompt_length
    )
    parser.add_argument("--logging_steps", type=int, default=Config.logging_steps)
    parser.add_argument("--eval_steps", type=int, default=Config.eval_steps)
    parser.add_argument("--save_steps", type=int, default=Config.save_steps)
    parser.add_argument(
        "--save_total_limit", type=int, default=Config.save_total_limit
    )
    parser.add_argument("--val_split", type=float, default=Config.val_split)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--lora_r", type=int, default=Config.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=Config.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=Config.lora_dropout)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=list(Config.lora_target_modules),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=Config.dataloader_num_workers,
    )
    parser.add_argument("--report_to", default=Config.report_to)
    parser.add_argument(
        "--verify_curriculum_batches",
        type=int,
        default=Config.verify_curriculum_batches,
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--non_strict_data_validation", action="store_true")

    args = parser.parse_args()

    cfg = Config(
        **{
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "no_bf16",
                "no_gradient_checkpointing",
                "non_strict_data_validation",
            }
        }
    )
    cfg.bf16 = not args.no_bf16
    cfg.gradient_checkpointing = not args.no_gradient_checkpointing
    cfg.strict_data_validation = not args.non_strict_data_validation
    cfg.lora_target_modules = tuple(args.lora_target_modules)

    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if not 0.0 < cfg.val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {cfg.val_split}")
    if cfg.max_prompt_length >= cfg.max_length:
        raise ValueError(
            "max_prompt_length must be smaller than max_length: "
            f"{cfg.max_prompt_length} >= {cfg.max_length}"
        )
    if cfg.beta <= 0:
        raise ValueError(f"beta must be positive, got {cfg.beta}")
    if cfg.learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be positive, got {cfg.learning_rate}"
        )
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")


# =============================================================================
# Data validation and loading
# =============================================================================

def read_jsonl(path: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise TypeError(
                    f"{path}, line {line_number}: expected JSON object, "
                    f"got {type(item).__name__}"
                )
            records.append(item)
            if limit is not None and len(records) >= limit:
                break
    return records


def index_unique_by_id(
    records: Iterable[dict[str, Any]], source_name: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []

    for item in records:
        if "id" not in item:
            raise KeyError(f"{source_name}: record without 'id': {item}")
        item_id = str(item["id"])
        if item_id in indexed:
            duplicates.append(item_id)
        else:
            indexed[item_id] = item

    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(
            f"{source_name}: duplicate IDs detected ({len(duplicates)}): {preview}"
        )
    return indexed


def parse_difficulty(value: Any, item_id: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{item_id}: boolean is not a valid difficulty")

    if isinstance(value, (int, np.integer)):
        level = int(value)
    elif isinstance(value, float) and value.is_integer():
        level = int(value)
    else:
        text = str(value).strip()
        match = re.search(r"(?:level\s*)?([1-4])\b", text, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(
                f"{item_id}: cannot parse difficulty from {value!r}; "
                "expected integer 1-4 or text such as 'Level 3'"
            )
        level = int(match.group(1))

    if level not in {1, 2, 3, 4}:
        raise ValueError(f"{item_id}: difficulty must be 1-4, got {level}")
    return level


def extract_prompt_text(prompt_item: dict[str, Any], item_id: str) -> str:
    conversations = prompt_item.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{item_id}: conversations must be a non-empty list")

    first_turn = conversations[0]
    if not isinstance(first_turn, dict):
        raise TypeError(f"{item_id}: conversations[0] must be an object")

    content = first_turn.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{item_id}: conversations[0].content is empty")
    return content.strip()


def count_constraints(prompt_item: dict[str, Any], item_id: str) -> int:
    constraints = prompt_item.get("constraints")
    if not isinstance(constraints, list):
        raise TypeError(
            f"{item_id}: constraints must be a list, got "
            f"{type(constraints).__name__}"
        )

    cleaned = [x for x in constraints if str(x).strip()]
    if len(cleaned) != len(constraints):
        raise ValueError(f"{item_id}: constraints contains empty entries")
    return len(cleaned)


def get_response(item: dict[str, Any], source_name: str, item_id: str) -> str:
    response = item.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{source_name} {item_id}: response is empty")
    return response.strip()


def format_prompt(tokenizer: Any, prompt_text: str) -> str:
    """
    Produce a prompt ending at the assistant-generation boundary.

    DPOTrainer receives:
      prompt = chat template through assistant header
      chosen/rejected = raw assistant completions
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback for tokenizers without a chat template.
    return prompt_text.rstrip() + "\n"


def load_dpo_pairs(
    cfg: Config,
    tokenizer: Any,
    max_records: Optional[int] = None,
) -> list[dict[str, Any]]:
    with open(cfg.prompts_path, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)

    if not isinstance(raw_prompts, list):
        raise TypeError(
            f"{cfg.prompts_path}: expected a JSON list, "
            f"got {type(raw_prompts).__name__}"
        )

    prompt_by_id = index_unique_by_id(raw_prompts, "prompts")
    accepted_by_id = index_unique_by_id(
        read_jsonl(cfg.accepted_path, max_records), "accepted"
    )
    rejected_by_id = index_unique_by_id(
        read_jsonl(cfg.rejected_path, max_records), "rejected"
    )

    accepted_ids = set(accepted_by_id)
    rejected_ids = set(rejected_by_id)

    missing_in_rejected = sorted(accepted_ids - rejected_ids)
    missing_in_accepted = sorted(rejected_ids - accepted_ids)

    if missing_in_rejected or missing_in_accepted:
        message = (
            f"Pair ID mismatch: missing in rejected={len(missing_in_rejected)}, "
            f"missing in accepted={len(missing_in_accepted)}"
        )
        if cfg.strict_data_validation:
            raise ValueError(
                message
                + f"\nMissing rejected preview: {missing_in_rejected[:10]}"
                + f"\nMissing accepted preview: {missing_in_accepted[:10]}"
            )
        print(f"WARNING: {message}; only common IDs will be used.")

    common_ids = accepted_ids & rejected_ids

    pairs: list[dict[str, Any]] = []
    missing_prompt = 0
    identical = 0
    invalid = 0

    # Preserve accepted-file insertion order without relying on rejected-file order.
    ordered_ids = [item_id for item_id in accepted_by_id if item_id in common_ids]

    for item_id in ordered_ids:
        if item_id not in prompt_by_id:
            missing_prompt += 1
            if cfg.strict_data_validation:
                raise KeyError(f"{item_id}: ID not found in {cfg.prompts_path}")
            continue

        try:
            prompt_item = prompt_by_id[item_id]
            prompt_text = extract_prompt_text(prompt_item, item_id)
            n_constraints = count_constraints(prompt_item, item_id)
            difficulty = parse_difficulty(
                prompt_item.get("difficulty"), item_id
            )
            chosen = get_response(
                accepted_by_id[item_id], "accepted", item_id
            )
            rejected = get_response(
                rejected_by_id[item_id], "rejected", item_id
            )
        except (KeyError, TypeError, ValueError) as exc:
            invalid += 1
            if cfg.strict_data_validation:
                raise
            print(f"WARNING: skipping invalid pair {item_id}: {exc}")
            continue

        if chosen == rejected:
            identical += 1
            if cfg.strict_data_validation:
                raise ValueError(
                    f"{item_id}: chosen and rejected responses are identical"
                )
            continue

        pairs.append(
            {
                "id": item_id,
                "prompt": format_prompt(tokenizer, prompt_text),
                "chosen": chosen,
                "rejected": rejected,
                "_n_constraints": n_constraints,
                "_difficulty_level": difficulty,
                "_curriculum_bucket": constraint_bucket(n_constraints),
            }
        )

    print(
        "Data load summary: "
        f"loaded={len(pairs):,}, "
        f"missing_prompt={missing_prompt}, "
        f"identical={identical}, invalid={invalid}"
    )

    if len(pairs) < 2:
        raise ValueError("Need at least two valid DPO pairs")
    return pairs


# =============================================================================
# Curriculum and split
# =============================================================================

def constraint_bucket(n_constraints: int) -> int:
    if n_constraints <= 3:
        return 0
    if n_constraints <= 6:
        return 1
    return 2


BUCKET_NAMES = {
    0: "<=3 constraints",
    1: "4-6 constraints",
    2: ">=7 constraints",
}


def curriculum_key(pair: dict[str, Any]) -> tuple[int, int]:
    return (
        int(pair["_curriculum_bucket"]),
        int(pair["_difficulty_level"]),
    )


def sort_curriculum(
    pairs: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """
    Shuffle first, then stable-sort.

    This preserves easy-to-hard ordering while randomizing examples that have
    exactly the same constraint bucket and difficulty.
    """
    ordered = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    ordered.sort(key=curriculum_key)
    assert_curriculum_monotonic(ordered)
    return ordered


def assert_curriculum_monotonic(pairs: list[dict[str, Any]]) -> None:
    keys = [curriculum_key(pair) for pair in pairs]
    for index in range(1, len(keys)):
        if keys[index] < keys[index - 1]:
            raise AssertionError(
                f"Curriculum order decreased at index {index}: "
                f"{keys[index - 1]} -> {keys[index]}"
            )


def stratified_train_val_split(
    pairs: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Stratify on the exact curriculum cell:
      (constraint-count bucket, difficulty level).

    The split happens before curriculum sorting. Singleton cells remain in train.
    """
    rng = random.Random(seed)
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for pair in pairs:
        cells[curriculum_key(pair)].append(pair)

    train_pairs: list[dict[str, Any]] = []
    val_pairs: list[dict[str, Any]] = []

    for key in sorted(cells):
        cell = list(cells[key])
        rng.shuffle(cell)

        if len(cell) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(len(cell) * val_ratio)))
            n_val = min(n_val, len(cell) - 1)

        val_pairs.extend(cell[:n_val])
        train_pairs.extend(cell[n_val:])

    if not val_pairs:
        # Defensive fallback for extremely tiny debug datasets.
        rng.shuffle(train_pairs)
        val_pairs.append(train_pairs.pop())

    train_pairs = sort_curriculum(train_pairs, seed)
    rng.shuffle(val_pairs)

    if set(pair["id"] for pair in train_pairs) & set(
        pair["id"] for pair in val_pairs
    ):
        raise AssertionError("Train/validation ID leakage detected")

    return train_pairs, val_pairs


def print_distribution(
    name: str, pairs: list[dict[str, Any]]
) -> None:
    counts = Counter(curriculum_key(pair) for pair in pairs)
    print(f"\n{name} distribution ({len(pairs):,} pairs)")
    print("-" * 58)
    for bucket in range(3):
        row = []
        for difficulty in range(1, 5):
            row.append(f"L{difficulty}={counts[(bucket, difficulty)]:,}")
        print(f"{BUCKET_NAMES[bucket]:>18}: " + " | ".join(row))


def strip_metadata(pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        }
        for pair in pairs
    ]


# =============================================================================
# Token-length diagnostics
# =============================================================================

def token_length(tokenizer: Any, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )


def print_token_diagnostics(
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    cfg: Config,
    sample_cap: int = 10_000,
) -> None:
    """
    Tokenize at most sample_cap examples for diagnostics.
    This does not alter the actual training examples.
    """
    rng = random.Random(cfg.seed)
    sample = list(pairs)
    if len(sample) > sample_cap:
        sample = rng.sample(sample, sample_cap)

    prompt_lengths = np.asarray(
        [token_length(tokenizer, pair["prompt"]) for pair in sample]
    )
    chosen_lengths = np.asarray(
        [token_length(tokenizer, pair["chosen"]) for pair in sample]
    )
    rejected_lengths = np.asarray(
        [token_length(tokenizer, pair["rejected"]) for pair in sample]
    )
    pair_max_total = prompt_lengths + np.maximum(
        chosen_lengths, rejected_lengths
    )

    def summarize(name: str, values: np.ndarray) -> None:
        percentiles = np.percentile(values, [50, 90, 95, 99])
        print(
            f"{name:>22}: "
            f"p50={percentiles[0]:.0f}, "
            f"p90={percentiles[1]:.0f}, "
            f"p95={percentiles[2]:.0f}, "
            f"p99={percentiles[3]:.0f}, "
            f"max={values.max():.0f}"
        )

    print(f"\nToken diagnostics (sample n={len(sample):,})")
    print("-" * 78)
    summarize("prompt", prompt_lengths)
    summarize("chosen completion", chosen_lengths)
    summarize("rejected completion", rejected_lengths)
    summarize("prompt + max(comp)", pair_max_total)

    prompt_over = int((prompt_lengths > cfg.max_prompt_length).sum())
    total_over = int((pair_max_total > cfg.max_length).sum())

    print(
        f"Prompt > max_prompt_length ({cfg.max_prompt_length}): "
        f"{prompt_over:,}/{len(sample):,} "
        f"({100 * prompt_over / len(sample):.2f}%)"
    )
    print(
        f"Pair > max_length ({cfg.max_length}): "
        f"{total_over:,}/{len(sample):,} "
        f"({100 * total_over / len(sample):.2f}%)"
    )

    if prompt_over:
        print(
            "WARNING: Some prompts exceed max_prompt_length. Hard/multi-constraint "
            "examples may lose constraints during truncation."
        )


# =============================================================================
# Model
# =============================================================================

def load_model_and_tokenizer(
    cfg: Config,
) -> tuple[Any, Any, bool]:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id"
            )
        tokenizer.pad_token = tokenizer.eos_token

    # Prompts are left padded; completions are handled by DPOTrainer.
    tokenizer.padding_side = "left"

    cuda_available = torch.cuda.is_available()
    use_bf16 = (
        cfg.bf16
        and cuda_available
        and torch.cuda.is_bf16_supported()
    )
    compute_dtype = torch.bfloat16 if use_bf16 else (
        torch.float16 if cuda_available else torch.float32
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(
        f"Loading model: dtype={compute_dtype}, "
        f"world_size={world_size}, local_rank={local_rank}"
    )

    # Do not use device_map="auto" in distributed Trainer jobs.
    # Every rank must load initialized pretrained weights.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    return model, tokenizer, use_bf16


# =============================================================================
# Trainer
# =============================================================================

class SequentialCurriculumDPOTrainer(DPOTrainer):
    """
    DPOTrainer that preserves the pre-sorted training dataset order.

    The flexible signature supports Transformers versions where
    _get_train_sampler takes either no dataset argument or one dataset argument.
    Accelerate handles distributed dataloader sharding after this sampler is
    created.
    """

    def _get_train_sampler(
        self,
        train_dataset: Optional[Dataset] = None,
        *args: Any,
        **kwargs: Any,
    ) -> SequentialSampler:
        dataset = train_dataset
        if dataset is None:
            dataset = self.train_dataset
        if dataset is None:
            raise ValueError("train_dataset is not available")
        return SequentialSampler(dataset)


def make_dpo_config(cfg: Config, checkpoint_dir: str, use_bf16: bool) -> DPOConfig:
    """
    Build DPOConfig while filtering version-specific arguments.

    This makes the script more tolerant of moderate TRL version differences.
    """
    kwargs: dict[str, Any] = {
        "output_dir": checkpoint_dir,
        "run_name": cfg.run_name,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "beta": cfg.beta,
        "loss_type": cfg.loss_type,
        "bf16": use_bf16,
        "fp16": torch.cuda.is_available() and not use_bf16,
        "logging_steps": 1 if cfg.debug else cfg.logging_steps,
        "save_steps": 1 if cfg.debug else cfg.save_steps,
        "eval_steps": 1 if cfg.debug else cfg.eval_steps,
        "save_strategy": "steps",
        "eval_strategy": "steps",
        "save_total_limit": cfg.save_total_limit,
        "max_length": cfg.max_length,
        "max_prompt_length": cfg.max_prompt_length,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "report_to": cfg.report_to,
        "max_steps": 2 if cfg.debug else -1,
        "load_best_model_at_end": False,
        "log_level": "info",
    }

    signature = inspect.signature(DPOConfig.__init__)
    accepted = set(signature.parameters)

    # Compatibility: older Transformers/TRL used evaluation_strategy.
    if "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    ignored = sorted(set(kwargs) - set(filtered))
    if ignored:
        print(
            "INFO: Current DPOConfig does not accept these arguments; ignored: "
            + ", ".join(ignored)
        )

    return DPOConfig(**filtered)


def make_trainer(
    model: Any,
    tokenizer: Any,
    dpo_config: DPOConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> SequentialCurriculumDPOTrainer:
    kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": dpo_config,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
    }

    signature = inspect.signature(DPOTrainer.__init__)
    accepted = set(signature.parameters)

    # New TRL uses processing_class; older TRL uses tokenizer.
    if "processing_class" in accepted:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in accepted:
        kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "Unsupported TRL DPOTrainer API: neither processing_class nor "
            "tokenizer is accepted"
        )

    return SequentialCurriculumDPOTrainer(**kwargs)


def verify_sampler_order(
    trainer: SequentialCurriculumDPOTrainer,
    train_pairs: list[dict[str, Any]],
    n_batches: int,
) -> None:
    """
    Verify the sampler itself is sequential without forcing expensive model
    tokenization or consuming the real training dataloader.
    """
    sampler = trainer._get_train_sampler(trainer.train_dataset)
    observed_indices = []
    for index in sampler:
        observed_indices.append(int(index))
        if len(observed_indices) >= max(n_batches, 1):
            break

    expected = list(range(len(observed_indices)))
    if observed_indices != expected:
        raise AssertionError(
            "Curriculum sampler is not sequential. "
            f"Observed={observed_indices}, expected={expected}"
        )

    preview_n = min(max(n_batches, 1), len(train_pairs))
    print(f"\nCurriculum sampler verified. First {preview_n} examples:")
    for index, pair in enumerate(train_pairs[:preview_n]):
        print(
            f"  index={index:>4} | "
            f"bucket={pair['_curriculum_bucket']} "
            f"({BUCKET_NAMES[pair['_curriculum_bucket']]}) | "
            f"difficulty=L{pair['_difficulty_level']} | "
            f"n_constraints={pair['_n_constraints']} | "
            f"id={pair['id']}"
        )

    print(f"Last {preview_n} examples:")
    start = len(train_pairs) - preview_n
    for offset, pair in enumerate(train_pairs[-preview_n:]):
        print(
            f"  index={start + offset:>4} | "
            f"bucket={pair['_curriculum_bucket']} "
            f"({BUCKET_NAMES[pair['_curriculum_bucket']]}) | "
            f"difficulty=L{pair['_difficulty_level']} | "
            f"n_constraints={pair['_n_constraints']} | "
            f"id={pair['id']}"
        )


# =============================================================================
# Saving and execution
# =============================================================================

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    root = Path(checkpoint_dir)
    if not root.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def save_adapters_and_tokenizer(
    trainer: DPOTrainer,
    tokenizer: Any,
    output_dir: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Saving final LoRA adapter and tokenizer to {output}")
    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))


def print_run_header(cfg: Config) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_global_batch = (
        cfg.per_device_train_batch_size
        * cfg.gradient_accumulation_steps
        * world_size
    )

    print("=" * 80)
    print("GPT-OSS-20B DPO — ordered curriculum training")
    print("=" * 80)
    print(f"transformers version : {transformers.__version__}")
    print(f"trl version          : {trl.__version__}")
    print(f"run name             : {cfg.run_name}")
    print(f"model                : {cfg.model_path}")
    print(f"accepted             : {cfg.accepted_path}")
    print(f"rejected             : {cfg.rejected_path}")
    print(f"prompts              : {cfg.prompts_path}")
    print(f"beta                 : {cfg.beta}")
    print(f"learning rate        : {cfg.learning_rate}")
    print(f"epochs               : {cfg.num_train_epochs}")
    print(f"world size           : {world_size}")
    print(f"effective global BS  : {effective_global_batch}")
    print(f"max length           : {cfg.max_length}")
    print(f"max prompt length    : {cfg.max_prompt_length}")
    print(f"LoRA rank/alpha      : {cfg.lora_r}/{cfg.lora_alpha}")
    print(f"debug                : {cfg.debug}")
    print(f"resume               : {cfg.resume}")
    print("=" * 80)


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    print_run_header(cfg)

    run_dir = Path(cfg.output_root) / cfg.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Model/tokenizer first because prompt formatting depends on chat_template.
    model, tokenizer, use_bf16 = load_model_and_tokenizer(cfg)

    max_records = 100 if cfg.debug else None
    pairs = load_dpo_pairs(
        cfg,
        tokenizer,
        max_records=max_records,
    )

    print_token_diagnostics(tokenizer, pairs, cfg)

    # Critical: split before sorting.
    train_pairs, val_pairs = stratified_train_val_split(
        pairs,
        val_ratio=cfg.val_split,
        seed=cfg.seed,
    )

    print_distribution("Train", train_pairs)
    print_distribution("Validation", val_pairs)
    assert_curriculum_monotonic(train_pairs)

    train_dataset = Dataset.from_list(strip_metadata(train_pairs))
    val_dataset = Dataset.from_list(strip_metadata(val_pairs))

    dpo_config = make_dpo_config(
        cfg,
        checkpoint_dir=str(checkpoint_dir),
        use_bf16=use_bf16,
    )
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        dpo_config=dpo_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    verify_sampler_order(
        trainer,
        train_pairs,
        n_batches=cfg.verify_curriculum_batches,
    )

    resume_checkpoint: Optional[str] = None
    if cfg.resume:
        resume_checkpoint = find_latest_checkpoint(str(checkpoint_dir))
        if resume_checkpoint is None:
            raise FileNotFoundError(
                f"--resume was requested, but no checkpoint was found under "
                f"{checkpoint_dir}"
            )
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    train_result = trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    save_adapters_and_tokenizer(
        trainer,
        tokenizer,
        cfg.best_model_dir,
    )

    config_path = run_dir / "resolved_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    print(f"Saved resolved config to {config_path}")


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
