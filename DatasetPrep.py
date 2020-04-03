import pandas as pd
import numpy as np

df = pd.read_csv("Diabetes.csv")
df.info()
df.head()

df2 = df[['Glucose', 'BMI', 'Age', 'Outcome']]
df2.head()
df2.describe()

mask = (df2[df2.columns[:-1]] == 0).any(axis = 1)
df3 = df2[~mask]

df3.groupby('Outcome').mean()

df3.groupby('Outcome').agg({'Glucose': 'mean', 'BMI': 'median', 'Age': 'sum'})

df3.groupby('Outcome').agg(['mean', 'median'])
positive = df3.loc[df3.Outcome == 1]
negative = df3.loc[df3.Outcome != 1]

# different sample sizes in each group - need to normlise for a lot of stats tests!
print(positive.shape, negative.shape)

df3.to_csv("clean_diabetes.csv", index = False)