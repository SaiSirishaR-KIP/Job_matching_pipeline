import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, Trainer, TrainingArguments, 
                          EarlyStoppingCallback, AutoConfig, AutoModel)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import torch.nn as nn

# ----------------------
# Data Loading & Preprocessing
# ----------------------
file_path = "/Volumes/Seagate Backup /project_job_matching/Data/merged_output.csv"  # File location
data = pd.read_csv(file_path)

# Ensure all job titles are strings
data['job_title'] = data['job_title'].astype(str)

# Create a combined input and encode labels
data['combined_input'] = data['job_skills'].astype(str) + " [SEP] " + data['search_position'].astype(str)
label_encoder = LabelEncoder()
data['job_title_encoded'] = label_encoder.fit_transform(data['job_title'])

# ----------------------
# Define Custom Dataset
# ----------------------
class JobSkillsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Create a mapping from label name to an integer ID (based on sorted unique job titles)
        self.label_to_id_map = {label: idx for idx, label in enumerate(sorted(data['job_title'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['combined_input']
        label = self.label_to_id_map[row['job_title']]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ----------------------
# Tokenizer & Dataset Creation
# ----------------------
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
max_length = 128
batch_size = 32  # Updated batch size

pytorch_dataset = JobSkillsDataset(data, tokenizer, max_length)

# Split dataset into training and testing sets
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataset, test_dataset = split_dataset(pytorch_dataset, train_ratio=0.8)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------
# Define the Custom Model
# ----------------------
class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Try to get the pooled output (if available) or use the first token embedding
        if isinstance(outputs, tuple):
            pooled_output = outputs[1]
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return loss, logits
        else:
            return logits

    def save(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))
        torch.save(self.state_dict(), os.path.join(save_directory, "custom_model.pt"))

    @classmethod
    def load(cls, load_directory, model_name, num_labels):
        model = cls(model_name, num_labels)
        model.model = AutoModel.from_pretrained(load_directory)
        model.classifier.load_state_dict(torch.load(os.path.join(load_directory, "classifier.pt")))
        model.load_state_dict(torch.load(os.path.join(load_directory, "custom_model.pt")))
        return model

# ----------------------
# Model Initialization & Optional Checkpoint Weight Loading
# ----------------------
config = AutoConfig.from_pretrained("xlm-roberta-base")
config.num_labels = len(label_encoder.classes_)
model = CustomModel("xlm-roberta-base", num_labels=config.num_labels)

# Check if GPU is available and move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Instead of resuming training with the old checkpoint (which includes optimizer state and batch size info),
# we load only the model weights (if the checkpoint exists) and then start training with the new data.
checkpoint_path = "./results/checkpoint-162050"
if os.path.isdir(checkpoint_path):
    checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.isfile(checkpoint_file):
        state_dict = torch.load(checkpoint_file, map_location=device)
        # strict=False allows missing keys (e.g., optimizer state) to be skipped
        model.load_state_dict(state_dict, strict=False)
        print("Loaded model weights from checkpoint:", checkpoint_path)
    else:
        print("No pytorch_model.bin found in checkpoint directory.")
else:
    print("Checkpoint directory not found. Training from scratch.")

# ----------------------
# Training Setup using Hugging Face Trainer
# ----------------------
training_args = TrainingArguments(
    output_dir='./results',              # output directory for checkpoints and logs
    num_train_epochs=5,                  # number of training epochs
    per_device_train_batch_size=batch_size,   # training batch size per device
    per_device_eval_batch_size=batch_size,    # evaluation batch size per device
    warmup_steps=500,                    # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                   # weight decay for optimization
    logging_dir='./logs',                # directory for storing logs
    logging_steps=10,
    eval_strategy="epoch",               # Evaluate at the end of each epoch
    save_strategy="epoch",               # Save checkpoint at the end of each epoch
    load_best_model_at_end=True,         # Automatically load best model at end of training
    metric_for_best_model="eval_loss",   # Metric to determine best model
    greater_is_better=False,             # Lower evaluation loss is better
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Train the model (note: we are not using resume_from_checkpoint here)
trainer.train()

# ----------------------
# Save the Fine-Tuned Model and Label Encoder
# ----------------------
model.save('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
with open('./label_encoder_classes.pkl', 'wb') as f:
    pickle.dump(label_encoder.classes_, f)

# ----------------------
# Prediction on Test Data and Save Job Recommendations
# ----------------------
model.eval()
predictions = []
true_labels = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs
    topk = torch.topk(logits, k=3, dim=1)
    predictions.extend(topk.indices.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Map IDs back to label names
id_to_label = {idx: label for idx, label in enumerate(label_encoder.classes_)}
decoded_predictions = [[id_to_label[idx] for idx in pred] for pred in predictions]

# Save the top 3 job recommendations for each test sample to a file
with open('job_recommendations.txt', 'w') as f:
    for i, preds in enumerate(decoded_predictions):
        f.write(f"Sample {i+1}:\n")
        f.write(f"True Label: {id_to_label[true_labels[i]]}\n")
        f.write(f"Top 3 Predictions: {', '.join(preds)}\n\n")

print("Job recommendations saved to job_recommendations.txt")
