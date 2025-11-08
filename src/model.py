import numpy as np
import pandas as pd 
import os
import torch
import time
import matplotlib.pyplot as plt
import librosa 
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

train_df = pd.read_csv("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/processed/train_features.csv")
valid_df = pd.read_csv("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/processed/val_features.csv")

y_train = train_df["labels"]
X_train = train_df.drop(columns = ["labels", "file_paths"])
y_valid = valid_df["labels"]
X_valid = valid_df.drop(columns = ["labels", "file_paths"])


# fixed random_state to ensure reproducibility
clf1 = LogisticRegression(random_state=42)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = RandomForestClassifier(random_state=42)
clf4 = KNeighborsClassifier(n_neighbors=42) 

classifiers = [clf1, clf2, clf3, clf4]
# classifiers = [clf2]

result = []
for i, c in enumerate(classifiers):
    pipe = Pipeline([
        # ('norm', Normalizer(norm='l2')),  # sample-wise norm
        ('clf', c)
    ])
    start_time = time.time()
    pipe.fit(X_train,y_train)
    y_pred = c.predict(X_valid)
    end_time = time.time()
    report = classification_report(y_valid, y_pred, output_dict = True)
    # print(report)
    acc = report['accuracy']
    result.append({
        'model' : f"clf{i+1}",
        'accuracy' : acc,
        'runtime' : end_time - start_time,
        'report' : report,
        'hyperparameters' : c.get_params
    })

# print(result)



### save plots comparing these three models
result_df = pd.DataFrame(result)

plt.bar(result_df["model"], result_df["accuracy"]) 
plt.ylabel("accuracy")
plt.title("model accuracy comparison")
plt.ylim(0, 1)
for i, v in enumerate(result_df['accuracy']):
   plt.text(i, v + 0.01, f"{v:.2f}")
plt.savefig("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/media/accuracy_comp.png")
plt.close()

plt.bar(result_df['model'], result_df['runtime'])
plt.ylabel("runtime in seconds")
plt.title("model training time comparison")
for i, v in enumerate(result_df['runtime']):
   plt.text(i, v + 0.01, f"{v:.2f}")
plt.savefig("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/media/runtime_comp.png")
plt.close()

# final model: random forest (cl3)
# clf3.fit(X_train,y_train)
y_pred = clf3.predict(X_valid)
report = classification_report(y_valid, y_pred, output_dict = False)
with open("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/media/random_forest_report.txt", "w") as f:
   f.write(report)
print(report)