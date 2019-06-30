import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
test2 = pd.read_csv("./test.csv")

y = []
for tt in train['Survived']:
    y.append(tt)
y = np.array(y, dtype=np.long)
all = pd.concat([train, test], ignore_index=True, sort=False)
all = all.loc[:, ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare', 'Embarked', 'Name']]
all['Age'] = all['Age'].fillna(all['Age'].mean())
all['Fare'] = all['Fare'].fillna(all['Fare'].mean())
all['Age'] = all['Age'] / 5

a = all['Name']
for i in range(len(a)):
    a[i] = a[i].split(', ')[1].split('. ')[0]

all = pd.get_dummies(all)

train_len = train.shape[0]
train = all[:train_len]
test = all[train_len:]
train = train.drop('PassengerId', axis=1)
test = test.drop('PassengerId', axis=1)

sc = RobustScaler()

x = train.to_numpy()
#x = sc.fit_transform(x)

clf = SVC(kernel='linear')
clf.fit(x, y)
test = test.fillna(0)

x = test.to_numpy()
#x = sc.transform(x)

predicted = clf.predict(x)

preds = []
mean = predicted.mean()
for p in predicted:
    if p > mean:
        preds.append(1)
    else:
        preds.append(0)

output = pd.DataFrame({'PassengerId': test2.PassengerId, 'Survived': preds})
output.to_csv('submission.csv', index=False)

a = pd.read_csv("./submission.csv")
b = pd.read_csv("./gender_submission.csv")

cnt = 0
for i in range(len(a['Survived'])):
    if a['Survived'][i] != b['Survived'][i]:
        cnt += 1
print(cnt)
