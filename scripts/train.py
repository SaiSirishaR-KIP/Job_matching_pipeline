import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import numpy as np
import pickle
import torch

# Load data
file_path = '../content/skills_title_position_10k.csv'  # File location
df = pd.read_csv(file_path)

# Preprocessing
df['combined_input'] = df['job_skills'].astype(str) + " [SEP] " + df['search_position'].astype(str)
label_encoder = LabelEncoder()
df['job_title_encoded'] = label_encoder.fit_transform(df['job_title'])

# Train-Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['combined_input'], df['job_title_encoded'], test_size=0.2, random_state=42
)

# Print the first few entries of train_texts and val_texts for debugging
print("First few entries of train_texts:", train_texts.head())
print("First few entries of val_texts:", val_texts.head())

# Load Tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the data
try:
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
except Exception as e:
    print(f"An error occurred during tokenization: {e}")
    train_encodings = None
    val_encodings = None

if train_encodings is not None and val_encodings is not None:
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=10,             # number of training epochs
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=8,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",     # Evaluate every epoch
        save_strategy="epoch",           # Save checkpoint every epoch
        load_best_model_at_end=True,     # Load the best model at the end of training
        metric_for_best_model="eval_loss", # Use evaluation loss to select the best model
        greater_is_better=False,         # Lower evaluation loss is better
        no_cuda=not torch.cuda.is_available()  # Use GPU if available
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Early stopping
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned Model and Label Encoder
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
    with open('./label_encoder_classes.pkl', 'wb') as f:
        pickle.dump(label_encoder.classes_, f)
else:
    print("Tokenization failed. Please check the input data and preprocessing steps.")
