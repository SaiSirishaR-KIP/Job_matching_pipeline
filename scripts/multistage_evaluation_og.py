import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os

#############################################
# 1. Define / Load the Custom Model
#############################################

class CustomModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(self.model.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # XLM-RoBERTa typically puts pooled output at index 1
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    @classmethod
    def load(cls, load_directory, model_name, num_labels):
        model = cls(model_name, num_labels)
        # Load base model weights
        model.model = AutoModel.from_pretrained(load_directory)
        # Load classifier head weights
        classifier_path = os.path.join(load_directory, "classifier.pt")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        # Load entire custom model state (incl. dropout, etc.)
        custom_path = os.path.join(load_directory, "custom_model.pt")
        model.load_state_dict(torch.load(custom_path, map_location="cpu"))
        return model

#############################################
# 2. (Optional) Helper Functions for Filtering
#############################################

def is_physical_job(job_title):
    """
    Basic heuristic or dictionary-based approach.
    Adjust logic as needed for your dataset.
    For example, 'nurse', 'construction', 'warehouse' might be considered physical.
    """
    job_title_lower = job_title.lower()
    physical_keywords = ["nurse", "construction", "warehouse", "maintenance", "fitness"]
    return any(keyword in job_title_lower for keyword in physical_keywords)

def is_extrovert_job(job_title):
    """
    Example: 'sales', 'public relations', 'marketing manager', etc. 
    Adjust logic to match your domain.
    """
    job_title_lower = job_title.lower()
    extrovert_keywords = ["sales", "public relations", "spokesperson", "marketing", "business development"]
    return any(keyword in job_title_lower for keyword in extrovert_keywords)

def filter_jobs_by_constraints(job_titles, constraints, personality):
    """
    Filter or re-rank the predicted job_titles (list of strings)
    based on constraints/personality. Example logic below:
    """
    if not isinstance(constraints, str):
        constraints = ""
    if not isinstance(personality, str):
        personality = ""

    filtered = []
    for title in job_titles:
        # Example logic: if user has health issues, skip physically demanding roles
        if "health" in constraints.lower():
            if is_physical_job(title):
                continue

        # Example logic: if user is an introvert, skip very extrovert-centric jobs
        if "introvert" in personality.lower():
            if is_extrovert_job(title):
                continue

        # Add other checks as you see fit
        filtered.append(title)

    # If no titles left after filtering, return original to avoid empty results
    if len(filtered) == 0:
        return job_titles
    return filtered

#############################################
# 3. Prepare a Dataset Class for Inference
#############################################

class UnlabeledJobSkillsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        data: a DataFrame that includes 'combined_input' column
        tokenizer: huggingface tokenizer
        max_length: int, max token length
        """
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

#############################################
# 4. Main Inference Pipeline
#############################################

def main():
    # -----------------------------
    # 4.1. Load the saved label encoder
    # -----------------------------
    label_encoder_path = "/Volumes/Seagate Backup /project_job_matching/label_encoder_classes.pkl"  # adjust path if needed
    with open(label_encoder_path, 'rb') as f:
        label_classes = pickle.load(f)
    id_to_label = {idx: label for idx, label in enumerate(label_classes)}

    # -----------------------------
    # 4.2. Load the tokenizer and model
    # -----------------------------
    model_dir = "/Volumes/Seagate Backup /project_job_matching/fine_tuned_model"  # the directory where you saved your trained model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_name = "xlm-roberta-base"   # same model used during training
    num_labels = len(label_classes)

    model = CustomModel.load(model_dir, model_name, num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # -----------------------------
    # 4.3. Load your new data with constraints/personality
    # -----------------------------
    new_data_file = "/Volumes/Extreme SSD/TalentBridge/Automation_for_data_fromPtoS/Data/Transfomed_files_forAWSexp/transformed_client_data.csv"
  # CSV with columns [skills, search_position, constraints, personality, ...]
    new_data = pd.read_csv(new_data_file)

    # Make sure columns exist. If they differ in your dataset, rename below:
    if 'job_skills' not in new_data.columns:
        raise ValueError("Missing 'skills' column in input CSV.")
    if 'search_position' not in new_data.columns:
        raise ValueError("Missing 'search_position' column in input CSV.")
    if 'constraints' not in new_data.columns:
        new_data['constraints'] = ""  # or handle differently
    if 'personality' not in new_data.columns:
        new_data['personality'] = ""  # or handle differently

    # -----------------------------
    # 4.4. Build the combined_input ONLY from skills + search_position
    # -----------------------------
    def build_model_input(skills_text, position_text):
        return f"{skills_text} [SEP] {position_text}"

    combined_texts = []
    for _, row in new_data.iterrows():
        skill_str = str(row['job_skills'])
        pos_str = str(row['search_position'])
        combined_text = build_model_input(skill_str, pos_str)
        combined_texts.append(combined_text)

    new_data['combined_input'] = combined_texts

    # -----------------------------
    # 4.5. Create a dataset and dataloader
    # -----------------------------
    max_length = 128
    inference_dataset = UnlabeledJobSkillsDataset(new_data, tokenizer, max_length=max_length)
    batch_size = 32
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # 4.6. Inference Loop + Multi-Stage Logic
    # -----------------------------
    all_final_titles = []
    model.eval()
    row_index = 0  # keep track of global row index across batches

    with torch.no_grad():
        for batch in inference_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Stage 2: Model gets only the skill-based text
            logits = model(input_ids, attention_mask=attention_mask)

            # Get top-3 predictions
            topk = torch.topk(logits, k=3, dim=1)
            top_indices = topk.indices.cpu().numpy()

            # Stage 3: Filter or re-rank based on constraints, personality
            for i in range(len(top_indices)):
                predicted_ids = top_indices[i]
                predicted_titles = [id_to_label[idx] for idx in predicted_ids]

                # Get user constraints/personality from new_data
                constraints = str(new_data.loc[row_index, "constraints"])
                personality = str(new_data.loc[row_index, "personality"])

                # Filter or re-rank
                final_titles = filter_jobs_by_constraints(
                    job_titles=predicted_titles,
                    constraints=constraints,
                    personality=personality
                )

                all_final_titles.append(final_titles)
                row_index += 1

    # -----------------------------
    # 4.7. Save final recommendations
    # -----------------------------
    # For demonstration, we’ll just store top1, top2, top3 in each row (post-filter).
    top1_list = []
    top2_list = []
    top3_list = []

    for titles in all_final_titles:
        top1_list.append(titles[0] if len(titles) > 0 else "")
        top2_list.append(titles[1] if len(titles) > 1 else "")
        top3_list.append(titles[2] if len(titles) > 2 else "")

    new_data['final_top1'] = top1_list
    new_data['final_top2'] = top2_list
    new_data['final_top3'] = top3_list

    # Optionally, create an explanation or summary for each row
    def explain_recommendations(row):
        cons = row['constraints']
        pers = row['personality']
        t1 = row['final_top1']
        t2 = row['final_top2']
        t3 = row['final_top3']

        if not t1 and not t2 and not t3:
            return f"No suitable roles found given constraints={cons}, personality={pers}."
        else:
            chosen = [x for x in [t1, t2, t3] if x]
            return f"Based on your constraints ({cons}) and personality ({pers}), we suggest: {', '.join(chosen)}."

    new_data['explanation'] = new_data.apply(explain_recommendations, axis=1)

    # Save output to Excel or CSV
    output_file = "./multi_stage_recommendations.xlsx"
    new_data.to_excel(output_file, index=False)
    print(f"Final recommendations saved to: {output_file}")

#############################################
# 5. Run if main
#############################################

if __name__ == "__main__":
    main()
