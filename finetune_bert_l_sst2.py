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

# Hyperparameters
MODEL_PATH = "bert-large-uncased"
DATASET_NAME = "glue"
TASK_NAME = "sst2"
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambda_cr = 0.1
lambda_dr = 0.01
lambda_svdr = 0.01

# Preprocess function
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

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
        # Convert inputs to a mutable dictionary
        inputs = {k: v for k, v in inputs.items()}
        
        # Extract labels and inputs
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        task_loss_fn = nn.CrossEntropyLoss()
        task_loss = task_loss_fn(logits, labels)

        # Regularization losses
        with torch.no_grad():
            pretrained_outputs = self.pretrained_model(input_ids, attention_mask=attention_mask)
            pretrained_logits = pretrained_outputs.logits

        cr_loss = consistency_regularization(pretrained_logits, logits)
        dr_loss = diversity_regularization(logits)
        svd_loss = svd_regularization(logits, k=5)

        # Total loss
        loss = task_loss + lambda_cr * cr_loss + lambda_dr * dr_loss + lambda_svdr * svd_loss

        # Return the computed loss
        return (loss, outputs) if return_outputs else loss

# Main function
if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(DEVICE)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(DEVICE)
    pretrained_model.eval()

    # Load dataset
    dataset = load_dataset(DATASET_NAME, TASK_NAME)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Preprocess dataset
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    train_dataset = train_dataset.remove_columns(["idx", "sentence"])
    val_dataset = val_dataset.remove_columns(["idx", "sentence"])
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        evaluation_strategy="epoch",
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
    print("Training with regularization...")
    trainer_with_reg.train()

    # Save model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Model saved to ./fine_tuned_model")

# test------------------------------------