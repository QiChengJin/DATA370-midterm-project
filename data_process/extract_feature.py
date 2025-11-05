import numpy as np
import pandas as pd 
import os
import torch

import librosa 
import soundfile as sf

from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare_dataset(root_dir, n_mfcc=40):

    file_paths = []
    labels = []
    mfcc_features = []

    sub_dir =os.path.join(root_dir, "real")
    for file in os.listdir(sub_dir):
        if file.endswith("wav"):
            file_paths.append(os.path.join(sub_dir, file))
            labels.append(1)  # 1 is for real voice, 0 is for fake voice
    sub_dir =os.path.join(root_dir, "fake")
    for file in os.listdir(sub_dir):
        if file.endswith("wav"):
            file_paths.append(os.path.join(sub_dir, file))
            labels.append(0)  # 1 is for real voice, 0 is for fake voice
    
        # mfcc_transform = torchaudio.transforms.MFCC(n_mfcc = n_mfcc)


    for file_path in tqdm(file_paths):
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc, axis=1) # average all frames for each mfcc feature, shape changes from (40, T) to 40
        mfcc_features.append(mfcc)
    
    df = pd.DataFrame(mfcc_features, columns=[f"mfcc_{i+1}" for i in range(n_mfcc)])
    df["labels"] = labels
    df["file_paths"] = file_paths

    return df



train_df = prepare_dataset("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/for-2sec/for-2seconds/training")
valid_df = prepare_dataset("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/for-2sec/for-2seconds/validation")
test_df = prepare_dataset("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/for-2sec/for-2seconds/testing")

train_df.to_csv("train_features.csv", index=False)
valid_df.to_csv("val_features.csv", index=False)
test_df.to_csv("test_features.csv", index=False)








    