# DATA37000 - Midterm Project -- Qicheng Jin

## Project Incentives  
Nowadays scams are everywhere, some attackers use computer aided or AI generated fake voice to steal money or data. Besides that, some verifications might include using voiceprint, and without fake voice detection, we will abuse unauthorized accesses.   To address these problems, I decided to build a fake voice detection model. 


## Dataset Description  
I found a dataset in Kaggle called “The Fake-or-Real (FoR) Dataset (deepfake audio)” which fits my project goals. This dataset consists of more than 10,000 voice samples, each sample is a 2 second audio of speech. There are two labels or two classes, one is “real”, that means the speech is from human, and the other one is “fake”, that means the speech is artificially generated.


```python
import pandas as pd
import os
import librosa
from tqdm import tqdm
import numpy as np
```

## Data Cleaning and Preprocessing

### 1. remove silent audios
I discovered that there are no silent audio, so no need to remove any audios so far


```python
def total_silent(root_dir):
    file_paths = []
    labels = []
   #  mfcc_features = []

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

    count = 0
    for file_path in tqdm(file_paths):
        y, sr = librosa.load(file_path)
        if y is None or len(y) == 0 or np.allclose(y, 0):
            count += 1
    return count


train_audio_count = total_silent("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/raw/for-2sec/for-2seconds/training")
valid_audio_count = total_silent("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/raw/for-2sec/for-2seconds/validation")
test_audio_count = total_silent("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/raw/for-2sec/for-2seconds/testing")

print(f"there are total of {train_audio_count + valid_audio_count + test_audio_count} silent audios that need to be removed")

```

    100%|██████████| 13956/13956 [00:30<00:00, 463.30it/s]
    100%|██████████| 2826/2826 [00:03<00:00, 941.54it/s]
    100%|██████████| 1088/1088 [00:01<00:00, 957.08it/s]

    there are total of 0 silent audios that need to be removed


    


### 2. convert audio data to tabular form with 50 features 
To prepare the data into tabular form that can use for numerical analysis and machine learning algorithms, I used a feature extraction algorithm called MFCC. MFCC (Mel Frequency Cepstral Coefficients) is used to represent the short term power spectrum of a sound. This feature is widely used in audio and speech tasks as it captures characteristics of sound that human perceives.   
After the data processing, now I have the dataset prepared in tabular form:   
a 17870 * 52 table (50 MFCCs, 1 label, 1 filpath)  

(the feature extraction code implementation can be found in src/extract_feature.py)


```python
df1 = pd.read_csv("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/processed/train_features.csv")
df2 = pd.read_csv("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/processed/val_features.csv")
df3 = pd.read_csv("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/data/processed/test_features.csv")

df = pd.concat([df1, df2, df3])
print(df.shape)
```

    (17870, 52)


## Exploratory Data Analysis

summary statistics for each MFCC feature


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mfcc_1</th>
      <th>mfcc_2</th>
      <th>mfcc_3</th>
      <th>mfcc_4</th>
      <th>mfcc_5</th>
      <th>mfcc_6</th>
      <th>mfcc_7</th>
      <th>mfcc_8</th>
      <th>mfcc_9</th>
      <th>mfcc_10</th>
      <th>...</th>
      <th>mfcc_42</th>
      <th>mfcc_43</th>
      <th>mfcc_44</th>
      <th>mfcc_45</th>
      <th>mfcc_46</th>
      <th>mfcc_47</th>
      <th>mfcc_48</th>
      <th>mfcc_49</th>
      <th>mfcc_50</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>...</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
      <td>17870.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-229.182395</td>
      <td>126.534384</td>
      <td>-26.443994</td>
      <td>44.927524</td>
      <td>-17.743433</td>
      <td>4.256712</td>
      <td>-16.725834</td>
      <td>-5.144449</td>
      <td>-14.292185</td>
      <td>-5.167326</td>
      <td>...</td>
      <td>0.860682</td>
      <td>-0.556844</td>
      <td>1.084792</td>
      <td>-0.670612</td>
      <td>1.143850</td>
      <td>-0.375415</td>
      <td>0.580759</td>
      <td>0.363031</td>
      <td>0.115251</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.652421</td>
      <td>26.161611</td>
      <td>17.321539</td>
      <td>18.146196</td>
      <td>15.883998</td>
      <td>17.561463</td>
      <td>13.424685</td>
      <td>12.041883</td>
      <td>9.023195</td>
      <td>7.067303</td>
      <td>...</td>
      <td>3.358527</td>
      <td>3.085604</td>
      <td>2.814101</td>
      <td>3.049707</td>
      <td>2.644111</td>
      <td>2.834018</td>
      <td>2.663453</td>
      <td>2.823404</td>
      <td>2.508522</td>
      <td>0.500014</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-503.617860</td>
      <td>14.726707</td>
      <td>-114.838320</td>
      <td>-23.143679</td>
      <td>-69.104100</td>
      <td>-48.750860</td>
      <td>-62.979073</td>
      <td>-52.653880</td>
      <td>-50.129410</td>
      <td>-38.848488</td>
      <td>...</td>
      <td>-10.705741</td>
      <td>-10.909400</td>
      <td>-10.866168</td>
      <td>-12.381920</td>
      <td>-8.151957</td>
      <td>-13.504985</td>
      <td>-10.382147</td>
      <td>-9.570374</td>
      <td>-7.839318</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-255.859332</td>
      <td>109.195960</td>
      <td>-36.560716</td>
      <td>32.910194</td>
      <td>-29.599772</td>
      <td>-9.046285</td>
      <td>-25.940466</td>
      <td>-15.087801</td>
      <td>-19.967878</td>
      <td>-9.940807</td>
      <td>...</td>
      <td>-1.475672</td>
      <td>-2.700977</td>
      <td>-0.882078</td>
      <td>-2.875594</td>
      <td>-0.726188</td>
      <td>-2.401941</td>
      <td>-1.217819</td>
      <td>-1.677310</td>
      <td>-1.577188</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-226.063365</td>
      <td>125.889067</td>
      <td>-25.566130</td>
      <td>45.954718</td>
      <td>-16.860660</td>
      <td>5.671578</td>
      <td>-16.709551</td>
      <td>-4.379413</td>
      <td>-14.303268</td>
      <td>-5.001461</td>
      <td>...</td>
      <td>0.364957</td>
      <td>-0.899426</td>
      <td>0.763304</td>
      <td>-0.926757</td>
      <td>0.862549</td>
      <td>-0.515908</td>
      <td>0.362069</td>
      <td>0.253686</td>
      <td>-0.135199</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-198.924545</td>
      <td>144.273770</td>
      <td>-15.722644</td>
      <td>57.539904</td>
      <td>-6.073112</td>
      <td>17.904367</td>
      <td>-7.998008</td>
      <td>4.444562</td>
      <td>-8.829849</td>
      <td>-0.421329</td>
      <td>...</td>
      <td>2.765410</td>
      <td>1.257526</td>
      <td>2.689994</td>
      <td>1.328817</td>
      <td>2.663340</td>
      <td>1.470788</td>
      <td>2.153338</td>
      <td>2.137647</td>
      <td>1.523089</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-31.026243</td>
      <td>220.989850</td>
      <td>54.580482</td>
      <td>116.172460</td>
      <td>49.210550</td>
      <td>59.130684</td>
      <td>30.210896</td>
      <td>34.762990</td>
      <td>27.162127</td>
      <td>20.787428</td>
      <td>...</td>
      <td>20.628798</td>
      <td>16.785997</td>
      <td>20.057116</td>
      <td>17.065094</td>
      <td>17.152262</td>
      <td>14.623589</td>
      <td>16.078850</td>
      <td>16.338438</td>
      <td>15.274864</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 51 columns</p>
</div>



check if the dataset is balanced with two classes (real vs fake)  
(1 is for real, 0 is for fake)  

The bar plot shows that they are balanced very well


```python
df["labels"].value_counts(normalize=True).plot(kind="bar")

```




    <Axes: xlabel='labels'>




    
![png](README_files/README_13_1.png)
    


Based on the distribution of first 5 MFCC features, each 


```python
import matplotlib.pyplot as plt
df.iloc[:, :5].hist(bins=30)
plt.show()
```


    
![png](README_files/README_15_0.png)
    


Based on the plot, we can view correlations among all 50 MFCC features and we discovered that first few MFCC features are not highly correlated, and for higher coefficients (about mfcc_25 - mfcc_50), close MFCC features are highly correlated. But overall, only moderate correlations among all features, and it's a good thing to show diversity of information 


```python
import seaborn as sns
corr = df.iloc[:, :-2].corr()
sns.heatmap(corr, cmap="coolwarm")

```




    <Axes: >




    
![png](README_files/README_17_1.png)
    


Based on these observations, MFCCS are pretty much normalized and not s0 correlated, so I don't need to do some data scaling after. 

## Machine Learning Models and Performance Comparison

I chose 4 models: logistic regression, decision tree, random forest, and KNN   
the model implementation can be found on src/model.py

## 1. visualization of test accuracy 

cl1: logistic regression, test_acc = 87%  

cl2: decision tree, test_acc = 93%  

cl3: random forest, test_acc = 99%  

cl4: KNN, test_acc = 96%


Based on the test accuracy, the random forest performs the best, achieving the highest accuracy. It's likely because it captures complex nonlinear relationships and is robust to overfitting since it's an ensumble model that basically consits of a bunch of decision trees.   

KNN also performs well, that shows local pattern recognition is working very well for the dataset.  

The decision tree has moderate performance and The logistic regression has the lowest accuracy. It suggests that the data-label relationship is not so linear. While logistic regression is the only linear model among all 4 models I chose, it struggles to fit the data.



```python
import matplotlib.image as mpimg
img = mpimg.imread("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/media/accuracy_comp.png")
plt.imshow(img)
plt.axis("off")
```




    (np.float64(-0.5), np.float64(639.5), np.float64(479.5), np.float64(-0.5))




    
![png](README_files/README_22_1.png)
    


## 2.visualization on model running time

cl1: logistic regression, run_time = 0.05   

cl2: decision tree, run_time = 1.00   

cl3: random forest, run_time = 6.74  

cl4: KNN, run_time = 0.16  


The random forest achieves the highest test accuracy, the trade off is that it runs the longest. This is expected, since it trains many trees and aggregates their results.  

The decision tree on the other hand runs faster since it trains a single tree. But it sacrifices the accuracy on test set, which means less robustness.  

KNN trains quickly as well, since we only store the positions of each training point, so not so much training happened. However, the real computational costs happen in prediction stage, since this is a large dataset, and we are going to find neighbors of the prediction target.

The Logistic regression trains the fastest, as its linear optimization process, however it sacrifices non-linearality, so runs fast but fits bad.


```python
img = mpimg.imread("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/media/runtime_comp.png")
plt.imshow(img)
plt.axis("off")
```




    (np.float64(-0.5), np.float64(639.5), np.float64(479.5), np.float64(-0.5))




    
![png](README_files/README_25_1.png)
    


Overall, I will pick random forest as my model: It performs the best, the run time is reasonable. Here is the classification report of random forest:

              precision    recall  f1-score   support

           0       0.99      0.98      0.99      1413
           1       0.98      0.99      0.99      1413

    accuracy                           0.99      2826
    macro avg      0.99      0.99      0.99      2826
    weighted avg   0.99      0.99      0.99      2826





```python
report = pd.read_table("/Users/mac/Desktop/DATA37000-25AU/DATA370-midterm-project/media/random_forest_report.txt")
print(report)
```

                 precision    recall  f1-score   support
    0             0       0.99      0.98      0.99   ...
    1             1       0.98      0.99      0.99   ...
    2      accuracy                           0.99   ...
    3     macro avg       0.99      0.99      0.99   ...
    4  weighted avg       0.99      0.99      0.99   ...


## Discussion
Right now, I successfully build a model that can tell apart real and fake voice recordings that has great applications in autheticity verification processes and fraud prevention. That also shows that traditional machine learning models can achieve strong performance with good features even without deep learning.   

Among the models I evaluated, random forest achieved the highest performace. KNN and decision trees strike a balance between training time and accuracy. Logistic regression trained the fastest but struggled to capture the nonlinear nature of MFCC features. This also highlight trade-off between model complexity, training cost and prediction. 

In a real-world deployment, for e,g, in the backend development, we can record every 2 sec audio segment in real time, and go through the same feature extraction pipeline for prediction.   

There are limitations too: with the evolution of AI, future synthetic audios may be more resemble to human and be much harder to detect. In the future, we might need to constantly retrain and update the model with new data or adopt some deep learning models. 


## References

Abdeldayem, M. (2023). *The Fake-or-Real (FoR) Dataset* [Dataset]. York University, Toronto, Canada.  
Licensed under GNU Lesser General Public License 3.0.  
Retrieved from [https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

