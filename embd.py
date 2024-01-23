import pandas as pd
import openai
import numpy as np

# Load your CSV file
file_path = './GG18 - Climate - Approved.csv'  # Replace with your file path
projects_df = pd.read_csv(file_path)

# Set your OpenAI API key
openai.api_key = ''  # Replace with your OpenAI API key

# Specify the model for embeddings
model = "text-embedding-ada-002"  # Example model, you can choose a different one

# Function to generate embeddings
def generate_embeddings(text):
    try:
        response = openai.Embedding.create(input=text, model=model)
        embedding = response['data'][0]['embedding']
        return np.array(embedding, dtype=float)  # Ensure conversion to a float array
    except Exception as e:
        print(f"Error in generating embedding: {e}")
        return None

# Generate embeddings for each project description
embeddings = [generate_embeddings(description) for description in projects_df['Description']]

# Filter out any None values if there were errors
embeddings = [e for e in embeddings if e is not None]

# Convert to a NumPy array and save
embeddings_array = np.array(embeddings)
np.save('project_embeddings.npy', embeddings_array)
