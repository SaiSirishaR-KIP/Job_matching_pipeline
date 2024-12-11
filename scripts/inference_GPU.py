import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pickle

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer
model_name = "xlm-roberta-base"  # Adjust to your chosen model
model_path = "./fine_tuned_model"  # Path to the saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)  # Move model to GPU if available

# Load the label encoder
with open('./label_encoder_classes.pkl', 'rb') as file:
    label_classes = pickle.load(file)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# Function for inference
def predict_job_title(job_skills, search_position, top_k=3):
    # Combine job skills and search position into one input text
    combined_input = f"{job_skills} [SEP] {search_position}"
    
    # Tokenize the input text
    inputs = tokenizer(
        combined_input,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU if available
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        top_k_indices = torch.topk(logits, top_k, dim=1).indices.squeeze().tolist()
    
    # Decode the labels
    predicted_job_titles = label_encoder.inverse_transform(top_k_indices)
    return predicted_job_titles

# Example usage
job_skills_example = "Interested in mobile care. Wants a job with good conditions and proper work-life balance."
search_position_example = "Mobile care worker"
predicted_titles = predict_job_title(job_skills_example, search_position_example)
print(f"Top 3 Predicted Job Titles: {predicted_titles}")
