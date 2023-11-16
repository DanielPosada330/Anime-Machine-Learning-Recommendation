import pandas as pd
import seaborn
import seaborn as sns
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics, model_selection

# Create variable name to hold dataset

url_anime = "https://raw.githubusercontent.com/DanielPosada330/Anime-Machine-Learning-Recommendation/main/anime.csv"
names = ["anime_id", "name", "genre", "type", "rating", "episodes", "members", "amount of ratings",
         "category"]

# Create dataframe to read csv file

df_anime = pd.read_csv(url_anime, names=names, on_bad_lines="skip", delimiter=",")

# Create model utilizing Logistic Regression

mylog_model = linear_model.LogisticRegression()

# Assign dependent and independent variables from datasets

# Dependent variable is rating of the anime
df_anime.dropna()
y = df_anime.values[:199, 8]

# Independent variables are amount of episodes, members and amount of ratings

X = (df_anime.values[:199, 5:7])

# Testing the datasets

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

# X_train = X_train.reshape(-1, 1)

# y_train = y_train.reshape(-1, 1)
# Train the model using .fit() method

mylog_model.fit(X_train, y_train)

# Utilize the trained model to make a prediction on anime ratings

y_pred = mylog_model.predict(X_test)

# Acquire accuracy score based on prediction

print(metrics.accuracy_score(y_test, y_pred))

# Add histogram visualization

sns.histplot(df_anime, x="amount of ratings", hue="category", kde=True, bins=30)
pyplot.show()

# Add scatter plot visualization

sns.lmplot(x="members", y="amount of ratings", data=df_anime, fit_reg=False, hue="category")
pyplot.show()

# Add boxplot visualization

sns.set(style="whitegrid")
sns.boxplot(data=df_anime, x="category", y="episodes")
pyplot.show()
