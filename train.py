import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

#Making one action pass (train) but making other fail (test)
#Logistics will return score < 0.5, 
# Reverting the model to use logistics regression

# This will result in error in score < 0.5

# score.yml (evaluate) -> fail
# train.yml (build-push) -> pass
# test.yml (pull-and-run) -> fail/not run
model = LogisticRegression().fit(X, y)
# model = GaussianNB().fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
