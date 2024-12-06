import pandas as pd
import time
from sklearn.model_selection import train_test_split
from joblib import load
import json

data = pd.read_csv('embeddings_with_topics.csv')
data['embeddings'] = data.apply(lambda row: [row[str(i)] for i in range(row.size-1)], axis=1)
embeddings = data['embeddings'].tolist()
labels = data['topic'].tolist()

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

print(len(X_test))
prediction_times = {}
for algo in ('knn', 'nb', 'svc', 'sgd'):
    model = load(algo + '.joblib')
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_times[algo] = time.time() - start_time
print(prediction_times)