"""Fine-tune Helsinki-NLP/opus-mt-ml-en on the project's parallel CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Helsinki-NLP/opus-mt-ml-en"


def load_parallel_sentences(path: Path) -> list[tuple[str, str]]:
    """Read non-empty Malayalam-English pairs from a CSV file."""
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"malayalam", "english"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must have CSV columns named 'malayalam' and 'english'."
            )

        for row_number, row in enumerate(reader, start=2):
            source = (row.get("malayalam") or "").strip()
            target = (row.get("english") or "").strip()
            if source and target:
                pairs.append((source, target))
            elif source or target:
                raise ValueError(
                    f"{path}:{row_number} has only one side of a translation pair."
                )

    if not pairs:
        raise ValueError(f"No complete sentence pairs found in {path}.")
    return pairs


class TranslationDataset:
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: Any,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        source, target = self.pairs[index]
        encoded = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
        )
        encoded["labels"] = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
        )["input_ids"]
        return encoded


def build_metrics(tokenizer: Any, sacrebleu_module: Any):
    def compute_metrics(prediction: Any) -> dict[str, float]:
        predictions, labels = prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = sacrebleu_module.corpus_bleu(
            [text.strip() for text in decoded_predictions],
            [[text.strip() for text in decoded_labels]],
        )
        return {"bleu": bleu.score}

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained Malayalam-to-English translation model."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=PROJECT_DIR / "data" / "train.csv",
        help="Training CSV with malayalam and english columns.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=PROJECT_DIR / "data" / "validation.csv",
        help="Validation CSV with malayalam and english columns.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "opus_mt_finetuned",
        help="Directory for the fine-tuned model and tokenizer.",
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=1000,
        help="Save a resumable checkpoint every this many training steps.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop after this many validation epochs without BLEU improvement.",
    )
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional small subset for a quick smoke test.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_model_snapshot(model: Any, tokenizer: Any, output_dir: Path, label: str) -> Path:
    snapshot_dir = output_dir / label
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(snapshot_dir))
    tokenizer.save_pretrained(str(snapshot_dir))
    return snapshot_dir


def main() -> None:
    args = parse_args()
    os.environ["USE_TF"] = "0"
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.learning_rate <= 0
        or args.checkpoint_steps <= 0
        or args.early_stopping_patience <= 0
    ):
        raise ValueError(
            "Epochs, batch size, learning rate, checkpoint steps, and "
            "early-stopping patience must be positive."
        )
    if args.max_source_length <= 0 or args.max_target_length <= 0:
        raise ValueError("Maximum sequence lengths must be positive.")

    try:
        import sacrebleu
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            TrainerCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install them with "
            "'python -m pip install -r requirements-training.txt'."
        ) from error

    train_pairs = load_parallel_sentences(args.train_file)
    validation_pairs = load_parallel_sentences(args.validation_file)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive.")
        train_pairs = train_pairs[: args.max_train_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": args.model,
        "train_file_sha256": file_sha256(args.train_file),
        "validation_file_sha256": file_sha256(args.validation_file),
        "max_train_samples": args.max_train_samples,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }
    summary_path = args.output_dir / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("run_config") != run_config:
            raise ValueError(
                f"{summary_path} belongs to a different model or dataset/config. "
                "Use a new --output-dir to keep the existing trained model safe."
            )
        completed_epochs = float(summary.get("completed_epochs", 0))
        if args.epochs <= completed_epochs:
            print(
                f"Training is already complete ({completed_epochs:g} epochs). "
                f"Saved model and tokenizer are in {args.output_dir}; "
                "no retraining needed. Increase --epochs to continue training."
            )
            print(f"Training history: {args.output_dir / 'training_history.json'}")
            return

    print(f"Training pairs: {len(train_pairs):,}")
    print(f"Validation pairs: {len(validation_pairs):,}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Loading pretrained model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    train_dataset = TranslationDataset(
        train_pairs, tokenizer, args.max_source_length, args.max_target_length
    )
    validation_dataset = TranslationDataset(
        validation_pairs, tokenizer, args.max_source_length, args.max_target_length
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=build_metrics(tokenizer, sacrebleu),
    )

    class SaveTrainingHistoryCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            write_json(
                Path(args.output_dir) / "training_history.json",
                state.log_history,
            )

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            write_json(
                Path(args.output_dir) / "training_history.json",
                state.log_history,
            )

    class StepCheckpointCallback(TrainerCallback):
        def on_step_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            if (
                state.global_step > 0
                and state.global_step % checkpoint_steps == 0
            ):
                control.should_save = True
            return control

    class SaveEpochModelCallback(TrainerCallback):
        def on_epoch_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> None:
            epoch_index = int((state.epoch or 0) + 1e-9)
            snapshot_dir = save_model_snapshot(
                kwargs.get("model") or trainer.model,
                tokenizer,
                Path(args.output_dir),
                f"epoch_{epoch_index:02d}",
            )
            print(f"Saved epoch {epoch_index} model weights to: {snapshot_dir}")

    checkpoint_steps = args.checkpoint_steps
    trainer.add_callback(StepCheckpointCallback())
    trainer.add_callback(SaveEpochModelCallback())
    trainer.add_callback(SaveTrainingHistoryCallback())
    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    )
    last_checkpoint = get_last_checkpoint(str(args.output_dir))
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    evaluation = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.state.save_to_json(str(args.output_dir / "trainer_state.json"))
    write_json(args.output_dir / "training_history.json", trainer.state.log_history)
    write_json(
        summary_path,
        {
            "run_config": run_config,
            "completed_epochs": float(trainer.state.epoch or args.epochs),
            "evaluation": evaluation,
        },
    )

    print(f"Saved fine-tuned model to: {args.output_dir}")
    print(f"Saved training history to: {args.output_dir / 'training_history.json'}")
    print(f"Saved trainer/checkpoint state to: {args.output_dir / 'trainer_state.json'}")
    print(f"Validation BLEU: {evaluation.get('eval_bleu', float('nan')):.2f}")


if __name__ == "__main__":
    main()
