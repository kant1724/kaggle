import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from sklearn.svm import SVC

import pandas as pd
pd.options.display.max_columns = 100

import numpy as np

data = pd.read_csv('./train.csv')

def get_combined_data():
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')

    train.drop(['Survived'], 1, inplace=True)
    # train 과 test combine
    combined = train.append(test)
    combined.reset_index(inplace=True)
    # index와 PassengerId 제거
    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)

    return combined

combined = get_combined_data()

titles = set()
for name in data['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}

def get_titles():
    # 이름에서 Title 추출
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    # Title 을 재정의함
    combined['Title'] = combined.Title.map(Title_Dictionary)
    return combined

combined = get_titles()

grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

print(grouped_median_train)

def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) &
        (grouped_median_train['Title'] == row['Title']) &
        (grouped_median_train['Pclass'] == row['Pclass'])
    )
    return grouped_median_train[condition]['Age'].values[0]

def process_age():
    global combined
    # Age가 빈값일 경우 채워넣기
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    return combined
combined = process_age()


def process_names():
    global combined
    # 이름 컬럼 제거
    combined.drop('Name', axis=1, inplace=True)

    # class 분류
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)
    combined.drop('Title', axis=1, inplace=True)

    return combined

combined = process_names()

def process_fares():
    global combined
    # Fare는 평균값 입력
    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
    return combined

combined = process_fares()

def process_embarked():
    global combined
    # 빈값일 경우 S
    combined.Embarked.fillna('S', inplace=True)

    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    return combined

combined = process_embarked()

train_cabin, test_cabin = set(), set()

for c in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')

for c in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')


def process_cabin():
    global combined
    # 빈값일 경우 'U'
    combined.Cabin.fillna('U', inplace=True)

    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)
    combined.drop('Cabin', axis=1, inplace=True)

    return combined

combined = process_cabin()
def process_sex():
    global combined

    # 남자는 1, 여자는 0
    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
    return combined

combined = process_sex()

def process_pclass():
    global combined

    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    combined = pd.concat([combined, pclass_dummies], axis=1)
    combined.drop('Pclass', axis=1, inplace=True)

    return combined

combined = process_pclass()
def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'

tickets = set()
for t in combined['Ticket']:
    tickets.add(cleanTicket(t))


def process_ticket():
    global combined

    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    return combined

combined = process_ticket()

def process_family():
    global combined

    # 가족을 정의
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    return combined

combined = process_family()

def recover_train_test_target():
    global combined

    targets = pd.read_csv('./train.csv', usecols=['Survived'])['Survived'].values
    train = combined.iloc[:891]
    test = combined.iloc[891:]

    return train, test, targets

test2 = pd.read_csv("./test.csv")

train, test, y = recover_train_test_target()
x = train.to_numpy()
clf = SVC(kernel='linear')
clf.fit(x, y)
test = test.fillna(0)

x = test.to_numpy()

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
