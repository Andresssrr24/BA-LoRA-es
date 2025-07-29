import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model  # Import LoRA utilities

# Hyperparameters
MODEL_PATH = "dccuchile/bert-base-spanish-wwm-uncased"
DATASET_NAME = "TASS_DATASET_POLARITY"
#TASK_NAME = "sst2"
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambda_cr = 0.1
lambda_dr = 0.01
lambda_svdr = 0.01

# Map labels
label2id = {'N': 0, 'P': 1, 'NEU': 2}

# Preprocess function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Regularization: Consistency Regularization
def consistency_regularization(pretrained_logits, fine_tuned_logits):
    pretrained_norm = pretrained_logits / pretrained_logits.norm(dim=1, keepdim=True)
    fine_tuned_norm = fine_tuned_logits / fine_tuned_logits.norm(dim=1, keepdim=True)
    return torch.mean((pretrained_norm - fine_tuned_norm).pow(2))

# Regularization: Diversity Regularization
def diversity_regularization(outputs):
    batch_size, dim = outputs.size()
    outputs_centered = outputs - outputs.mean(dim=0)
    covariance_matrix = (outputs_centered.T @ outputs_centered) / (batch_size - 1)
    off_diagonal_elements = covariance_matrix.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
    return torch.sum(off_diagonal_elements.pow(2)) / dim

# Regularization: SVD Regularization
def svd_regularization(outputs, k=5):
    u, s, v = torch.svd(outputs)
    top_k_singular_values = s[:k]
    total_singular_values = torch.sum(s)
    return -torch.sum(top_k_singular_values) / total_singular_values

# Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, pretrained_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = pretrained_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"].to(self.pretrained_model.device)
        attention_mask = inputs["attention_mask"].to(self.pretrained_model.device)

        outputs = model(**inputs)
        logits = outputs.logits
        task_loss = outputs.loss

        with torch.no_grad():
            pretrained_outputs = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pretrained_logits = pretrained_outputs.logits

        cr_loss = consistency_regularization(pretrained_logits, logits)
        dr_loss = diversity_regularization(logits)
        svd_loss = svd_regularization(logits, k=5)

        loss = task_loss + lambda_cr * cr_loss + lambda_dr * dr_loss + lambda_svdr * svd_loss

        return (loss, outputs) if return_outputs else loss

# Main function
if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3).to(DEVICE) # 3 labels P, NEU, N

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Scaling factor
        target_modules=["query", "value"],  # Apply LoRA to specific layers
        lora_dropout=0.1,  # Dropout probability
        bias="none"  # LoRA doesn't modify bias terms
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Load pretrained model for consistency regularization
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3).to(DEVICE)
    pretrained_model.eval()

    # Load dataset
    dataset = load_dataset('csv', data_files={
        'train': f'/content/{DATASET_NAME}/tass_train_dev/train.tsv',
        'validation': f'/content/{DATASET_NAME}/tass_train_dev/dev.tsv'
        }, delimiter="\t" )

    # map labels to numbers
    def encode_labels(example):
        example['label'] = label2id[example['label']]
        return example
    dataset = dataset.map(encode_labels)    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Preprocess dataset
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    train_dataset = train_dataset.remove_columns(["tweet_id", "text"])
    val_dataset = val_dataset.remove_columns(["tweet_id", "text"])
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        metric_for_best_model='runtime',
    )

    # Trainer with regularization
    trainer_with_reg = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        pretrained_model=pretrained_model,
    )

    # Start training
    print("Training with LoRA and regularization...")
    trainer_with_reg.train()

    # Save model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Model saved to ./fine_tuned_model")
