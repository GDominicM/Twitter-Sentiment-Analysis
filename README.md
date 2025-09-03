## Pipeline Overview  

### 1. Load Dataset  
- I put `  df = pd.read_csv(DATASET, header=None, encoding="latin-1")`   and `  print(df.head())`  
- It was able to print out the dataset

### 2. Keep Only Positive & Negative  
- Neutral tweets were removed using `  df = df[df["target"] != "2"]`         
- Postive and negative tweets were remapped using df["target"] = `df["target"].map({"0": 0, "4": 1, "1": 1}) ` 

### 3. Preprocess Text  
- I added a function to convert text to lowercase 

### 4. Train-Test Split  
- I split the data into train and test data and added:
` print(len(X_train))` 
` print(len(y_test))` 
- This resulted in:
Training data: 80927 and Test data : 20232

### 5. TF-IDF Vectorization  
- I added a vectorizer `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` 
 

### 6. Train Models  
- I trained 3 classifiers and printed out the accuraccy scores:  

| Model                | Accuracy |
|-----------------------|----------|
| BernoulliNB           | **0.7633946223803875** |
| Linear SVC            | **0.7834618426255437** |
| Logistic Regression   | **0.7865262949782523** |


### 7. Inference  
- I tested the models using diferent sentences and phrases 
- Based on these examples, the models rely on certain phrases or keywords to identify something as positive or negative. The models also cannot correctly identify somewhat complex phrases/sentences.

## Tests  

| Text                                            | BNB            | SVC            | LR             |
|-------------------------------------------------|----------------|----------------|----------------|
| *I love sisg!*                                  | Positive (89.20%) | Positive (79.73%) | Positive (98.28%) |
| *Hey that looks good! Actually it looks like trash! You suck! :)* | Positive (86.04%) | Positive (51.65%) | Positive (64.34%) |
| *Damn that was pretty nice.*                    | Positive (82.16%) | Positive (63.89%) | Positive (77.70%) |
| *He was really bad*                             | Negative (2.25%)  | Negative (27.82%) | Negative (11.96%) |
| *Life kinda hard rn*                            | Positive (53.08%) | Positive (50.54%) | Negative (48.60%) |
| *Huawei sucks compared to Samsung*              | Negative (18.16%) | Negative (18.62%) | Negative (1.67%)  |
| *I am going to kill someone*                    | Negative (14.62%) | Negative (37.33%) | Negative (25.42%) |

