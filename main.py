import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics, model_selection

## Create variable names to hold datasets

url_anime = "https://raw.githubusercontent.com/DanielPosada330/Anime-Machine-Learning-Recommendation/main/anime.csv"

url_ratings = "https://raw.githubusercontent.com/DanielPosada330/Anime-Machine-Learning-Recommendation/main/rating.csv"

## Create dataframe to read csv file

df_anime = pd.read_csv(url_anime, on_bad_lines="skip", delimiter=",")

## Create model utilizing Logisitic Regression

mylog_model = linear_model.LogisticRegression()

## Assign dependent and indpendent variables from datasets

y = df_anime.values[:, 5]

X = (df_anime.values[:, 10])

## Testing the datasets

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

## Train the model using .fit() method

mylog_model.fit(X, y)

## Utilize the trained model to make a prediction on anime ratings

y_pred = mylog_model.predict(X)

## Acquire accuracy score based on prediction

print(metrics.accuracy_score(y_test, y_pred))
