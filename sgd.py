import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from joblib import dump
import json

data = pd.read_csv('/home/herb/vk-project/embeddings_with_topics.csv')
data['embeddings'] = data.apply(lambda row: [row[str(i)] for i in range(row.size-1)], axis=1)
embeddings = data['embeddings'].tolist()
labels = data['topic'].tolist()

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

sgd_classifier = SGDClassifier(loss='log_loss', max_iter=1000, n_jobs=-1)  # You can choose different loss functions
start_time = time.time()
sgd_classifier.fit(X_train, y_train)
training_time = time.time() - start_time
dump(sgd_classifier, 'sgd_classifier.joblib')

y_pred = sgd_classifier.predict(X_test)

metrics = classification_report(y_test, y_pred, output_dict=True)
metrics['Training time'] = training_time 
print(metrics)
with open('sgd-metrics.json', 'w') as f:
    json.dump(metrics, f)   