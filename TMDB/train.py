import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as snsㄴ
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LassoCV
import json
import datetime

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
test2 = pd.read_csv("./test.csv")

y = []
y = np.log(train['revenue']).to_numpy()
all = pd.concat([train, test], ignore_index=True, sort=False)

all = all.loc[:, ['id', 'budget', 'spoken_languages', 'original_language', 'popularity', 'runtime', 'production_companies', 'production_countries', 'release_date', 'genres']]

mean = all['budget'].mean() / 100
all['budget'] = all['budget'].replace(0, mean)
all['popularity'] = all['popularity'].fillna(all['popularity'].mean())
all['runtime'] = all['runtime'].fillna(all['runtime'].mean())
all['genres'] = all['genres'].fillna('[{}]')
all['spoken_languages'] = all['spoken_languages'].fillna('[{}]')
all['production_companies'] = all['production_companies'].fillna('[{}]')
all['production_countries'] = all['production_countries'].fillna('[{}]')
all['release_date'] = all['release_date'].fillna('1/1/02')

a = all['release_date'].values[:]
for i in range(len(a)):
    d = int(a[i].split('/')[2])
    if int(d) > 19:
        d = int(d) - 100
    d += 100
    a[i] = d / 10

a = all['production_countries'].values[:]
for i in range(len(a)):
    a[i] = eval(a[i])[0].get('iso_3166_1', 'UNK')

a = all['production_companies'].values[:]
for i in range(len(a)):
    a[i] = eval(a[i])[0].get('name', 'UNK')

a = all['spoken_languages'].values[:]
for i in range(len(a)):
    a[i] = eval(a[i])[0].get('iso_639_1', 'UNK')

a = all['genres'].values[:]
for i in range(len(a)):
    a[i] = eval(a[i])[0].get('name', 'UNK')

all = pd.get_dummies(all)
sc = RobustScaler()
train_len = train.shape[0]
train = all[:train_len]
test = all[train_len:]
train.drop('id', axis=1)
x = sc.fit_transform(train)

model = Lasso(alpha=0.005, random_state=1)
model.fit(x, y)

test.drop('id', axis=1)
x = sc.transform(test)

pred = model.predict(x)
preds = np.exp(pred)
output = pd.DataFrame({'Id': test2.id, 'revenue': preds})
output.to_csv('submission.csv', index=False)
