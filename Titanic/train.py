import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn

input_size = 7
num_classes = 2
num_epochs = 10000
batch_size = 100
learning_rate = 0.004

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
test2 = pd.read_csv("./test.csv")

y = []
y = train['Survived'].to_numpy()
all = pd.concat([train, test], ignore_index=True, sort=False)
all = all.loc[:, ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare']]
all['Age'] = all['Age'].fillna(20)
all['Age'] = all['Age'] / 100
all['Fare'] = all['Fare'] / 10

all = pd.get_dummies(all)
train_len = train.shape[0]
print(train_len)
train = all[:train_len]
test = all[train_len:]
train = train.drop('PassengerId', axis=1)
test = test.drop('PassengerId', axis=1)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

x = train.to_numpy()

# Train the model
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x.astype(np.float32))
    targets = torch.from_numpy(y)

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
    preds.append(np.argmax(p))

output = pd.DataFrame({'PassengerId': test2.PassengerId, 'Survived': preds})
output.to_csv('submission.csv', index=False)
