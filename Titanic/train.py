import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn

input_size = 6
num_classes = 1
num_epochs = 10000
batch_size = 100
learning_rate = 0.001

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
test2 = pd.read_csv("./test.csv")

y = []
for tt in train['Survived']:
    y.append([tt])
y = np.array(y, dtype=np.long)
all = pd.concat([train, test], ignore_index=True, sort=False)
all = all.loc[:, ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare']]
all['Age'] = all['Age'].fillna(20)
all['Age'] = all['Age'] / 10
all['Fare'] = all['Fare'] / 100

a = all['Sex']
for i in range(len(a)):
    if a[i] == 'male':
        a[i] = 0.5
    else:
        a[i] = 1

all.to_csv('./dd.csv')

train_len = train.shape[0]
train = all[:train_len]
test = all[train_len:]
train = train.drop('PassengerId', axis=1)
test = test.drop('PassengerId', axis=1)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

x = train.to_numpy()

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
predicted = model(torch.from_numpy(x.astype(np.float32))).detach().numpy()
preds = []
for p in predicted:
    if p[0] > 0.3:
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
