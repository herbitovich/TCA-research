import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import SGDClassifier
#from sklearn.svm import SVC
#from sklearn.metrics import classification_report
from joblib import dump
from multiprocessing import Pool

# Step 1: Load the dataset
data = pd.read_csv('/home/herb/vk-project/articles_lemmatized_no_SW.csv')  # Replace with your actual CSV file path
data['text'] = data['headline'] + ' ' + data['body']
# Step 2: Initialize the BERT tokenizer and model
device = ('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
model = model.to(device)
def get_bert_embeddings_batch(batch_texts):
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    # Print how many rows have been vectorized in this batch
    print(f"Vectorized {len(batch_texts)} rows.")
    
    return embeddings

# Function to get BERT embeddings using multiprocessing
def get_bert_embeddings(texts, batch_size=16):
    all_embeddings = []
    # Create batches of texts
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Use multiprocessing to process batches in parallel
    with Pool(processes=8) as pool:  # Adjust the number of processes as needed
        results = pool.map(get_bert_embeddings_batch, batches)
    
    # Combine the results
    all_embeddings = np.vstack(results)
    
    # Print the total number of rows vectorized
    print(f"Total rows vectorized: {len(texts)}")
    
    return all_embeddings

# Step 4: Get embeddings for the text data
texts = []
labels = []
#iterate through all rows and add the row's text and label to the corresponding list only if the type(text) == str
for index, row in data.iterrows():
    if isinstance(row['text'], str) and isinstance(row['topic'], str):
        texts.append(row['text'])
        labels.append(row['topic'])

print("Rows: ", len(texts))
print("Started the embedding process")
embeddings = get_bert_embeddings(texts)
embeddings_df = pd.DataFrame(embeddings)

# Step 2: Add the 'topic' column to the embeddings DataFrame
embeddings_df['topic'] = labels  # Ensure the lengths match

# Step 3: Save the DataFrame to a CSV file
embeddings_df.to_csv('embeddings_with_topics.csv', index=False)
print("Embeddings saved to 'embeddings_with_topics.csv'")
