import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from transformers import pipeline

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

# Function to process DataFrame
def process_dataframe(df):
    combined_details = ' [SEP] '.join(df['Details'].tolist())
    return combined_details

# Function for inference
def predict_job_title(combined_details):
    # Tokenize the input text
    inputs = tokenizer(
        [combined_details],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU if available
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        top_index = torch.argmax(logits, dim=1).item()
    
    # Decode the label
    predicted_job_title = label_encoder.inverse_transform([top_index])[0]
    return predicted_job_title

# Example usage
file_path = "../Data/Nora_Bouanani_8911.xlsx"  # Excel file
client_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract client name from file name
df = pd.read_excel(file_path)
combined_details = process_dataframe(df)
predicted_title = predict_job_title(combined_details)

print(f"Job Role Recommendation for {client_name}:")
print(f"  Predicted Job Title: {predicted_title}")

# Function to summarize client profile
def summarize_profile(df):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    profile_text = ""
    for column in df.columns:
        profile_text += f"{column}: {df[column].astype(str).str.cat(sep=' ')} "
    summary = summarizer(profile_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Summarize the client's profile
client_summary = summarize_profile(df)

# Print the client summary
print(f"Client Profile Summary for {client_name}:")
print(f"  {client_summary}")

# Function for job recommendation based on summary
def recommend_job(summary):
    inputs = tokenizer(
        [summary],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU if available
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        top_index = torch.argmax(logits, dim=1).item()
    
    # Decode the label
    predicted_job_title = label_encoder.inverse_transform([top_index])[0]
    return predicted_job_title

# Get job recommendation based on the summary
predicted_title_from_summary = recommend_job(client_summary)

print(f"Job Role Recommendation based on Profile Summary for {client_name}:")
print(f"  Predicted Job Title: {predicted_title_from_summary}")