import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pickle
from transformers import AutoModel

# Load the pre-trained model and tokenizer
class CustomModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(self.model.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    @classmethod
    def load(cls, load_directory, model_name, num_labels):
        model = cls(model_name, num_labels)
        model.model = AutoModel.from_pretrained(load_directory)
        model.classifier.load_state_dict(torch.load(f"{load_directory}/classifier.pt"))
        model.load_state_dict(torch.load(f"{load_directory}/custom_model.pt"))
        return model

# Load the label encoder
with open('./label_encoder_classes.pkl', 'rb') as f:
    label_classes = pickle.load(f)

# Map labels
id_to_label = {idx: label for idx, label in enumerate(label_classes)}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
model = CustomModel.load('./fine_tuned_model', "xlm-roberta-base", num_labels=len(label_classes))

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Load the new unlabeled dataset
new_data_file = "../Data/transformed_client_data.csv"  # Replace with actual path
new_data = pd.read_csv(new_data_file)

# Preprocess the input
new_data['combined_input'] = new_data['job_skills'].astype(str) + " [SEP] " + new_data['search_position'].astype(str)

# Define dataset class for inference
class UnlabeledJobSkillsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['combined_input']
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# Create dataset and dataloader
unlabeled_dataset = UnlabeledJobSkillsDataset(new_data, tokenizer, max_length=128)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)

# Inference
predictions = []
with torch.no_grad():
    for batch in unlabeled_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask=attention_mask)
        topk = torch.topk(logits, k=3, dim=1)
        predictions.extend(topk.indices.cpu().numpy())

# Decode predictions
decoded_predictions = [[id_to_label[idx] for idx in pred] for pred in predictions]

# Save predictions to an Excel file
output_file = "../Data/unlabeled_data_predictions.xlsx"
output_data = {
    "file_name": new_data['file_name'],
    "top1": [preds[0] for preds in decoded_predictions],
    "top2": [preds[1] for preds in decoded_predictions],
    "top3": [preds[2] for preds in decoded_predictions],
    "ground_truth_info": new_data['job_skills']
}
output_df = pd.DataFrame(output_data)
output_df.to_excel(output_file, index=False)

print(f"Predictions saved to {output_file}")

