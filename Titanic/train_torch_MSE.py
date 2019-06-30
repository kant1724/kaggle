import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn

input_size = 34
num_classes = 1
num_epochs = 50000
batch_size = 100
learning_rate = 0.0003

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
test2 = pd.read_csv("./test.csv")

y = []
for tt in train['Survived']:
    y.append([tt])
y = np.array(y, dtype=np.long)
all = pd.concat([train, test], ignore_index=True, sort=False)
all = all.loc[:, ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare', 'Embarked', 'Name']]
all['Age'] = all['Age'].fillna(20)
a = all['Age']
for i in range(len(a)):
    if a[i] <= 10:
        a[i] = 'a'
    elif a[i] <= 20:
        a[i] = 'b'
    elif a[i] <= 30:
        a[i] = 'c'
    elif a[i] <= 40:
        a[i] = 'd'
    elif a[i] <= 50:
        a[i] = 'e'
    elif a[i] <= 60:
        a[i] = 'f'
    else:
        a[i] = 'g'

a = all['Name']
for i in range(len(a)):
    a[i] = a[i].split(', ')[1].split('. ')[0]

all['Fare'] = all['Fare'] / 100

all.to_csv('./table.csv')

all = pd.get_dummies(all)

train_len = train.shape[0]
train = all[:train_len]
test = all[train_len:]
train = train.drop('PassengerId', axis=1)
test = test.drop('PassengerId', axis=1)

sc = RobustScaler()

# Logistic regression model
model = nn.Linear(input_size, num_classes)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

x = train.to_numpy()
#x = sc.fit_transform(x)

# Train the model
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x.astype(np.float32))
    targets = torch.from_numpy(y.astype(np.float32))

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = test.to_numpy()
#x = sc.transform(x)

predicted = model(torch.from_numpy(x.astype(np.float32))).detach().numpy()
preds = []

for p in predicted:
    if p[0] > 0.48:
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
