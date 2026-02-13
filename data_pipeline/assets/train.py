import math
import os

from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch

from data_pipeline.resources import HuggingFaceConfig


@asset(group_name="training", deps=["train_val_splits"])
def trained_model(
    context: AssetExecutionContext,
    hf_config: HuggingFaceConfig,
) -> MaterializeResult:
    """Fine-tune the model using LoRA/QLoRA and evaluate on the validation set."""
    output_dir = hf_config.output_dir
    train_ds = load_from_disk(os.path.join(output_dir, "train"))
    val_ds = load_from_disk(os.path.join(output_dir, "val"))
    context.log.info(
        f"Loaded {len(train_ds)} train / {len(val_ds)} val examples"
    )

    # Quantization config for QLoRA (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    context.log.info(f"Loading model: {hf_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(hf_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA config
    lora_config = LoraConfig(
        r=hf_config.lora_r,
        lora_alpha=hf_config.lora_alpha,
        lora_dropout=hf_config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    context.log.info(
        f"LoRA trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # Training
    adapter_path = os.path.join(output_dir, "lora_adapter")
    training_args = TrainingArguments(
        output_dir=adapter_path,
        num_train_epochs=hf_config.num_train_epochs,
        per_device_train_batch_size=hf_config.per_device_train_batch_size,
        per_device_eval_batch_size=hf_config.per_device_train_batch_size,
        learning_rate=hf_config.learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=hf_config.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    context.log.info("Starting training...")
    train_result = trainer.train()
    train_loss = train_result.metrics["train_loss"]
    context.log.info(f"Training complete. Train loss: {train_loss:.4f}")

    # Evaluation
    context.log.info("Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    val_loss = eval_metrics["eval_loss"]
    val_perplexity = math.exp(val_loss)
    context.log.info(f"Val loss: {val_loss:.4f}, Val perplexity: {val_perplexity:.2f}")

    # Save adapter
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    context.log.info(f"Saved LoRA adapter to {adapter_path}")

    return MaterializeResult(
        metadata={
            "train_loss": MetadataValue.float(train_loss),
            "val_loss": MetadataValue.float(val_loss),
            "val_perplexity": MetadataValue.float(val_perplexity),
            "trainable_params": MetadataValue.int(trainable),
            "total_params": MetadataValue.int(total),
            "adapter_path": MetadataValue.path(adapter_path),
        }
    )
