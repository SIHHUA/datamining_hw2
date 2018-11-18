import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
from sklearn.metrics import accuracy_score

data = ["sex", "weight", "hieght", "visceral_fat", "body_fat", "obesity", "status"]
a = np.zeros([500, 7])
# 給值
for i in range(len(a)):
    a[i] = [random.randrange(0, 2, 1), random.randrange(30, 151, 1), random.randrange(100, 201, 1),
            random.randrange(1, 16, 1), random.randrange(1, 46, 1), random.randrange(-20, 21, 1), 0]
# 給答案
for i in range(len(a)):
    little = 0
    normal = 0
    excessive = 0
    if a[i][0] == 0:  # man
        # visceral_fat
        if a[i][3] < 4:
            little += 1
        elif a[i][3] > 6:
            excessive += 1
        else:
            normal += 1

        # body_fat
        if a[i][4] < 14:
            little += 1
        elif a[i][4] > 20:
            excessive += 1
        else:
            normal += 1
    else:  # woman
        # visceral_fat
        if a[i][3] < 2:
            little += 1
        elif a[i][3] > 4:
            excessive += 1
        else:
            normal += 1

        # body_fat
        if a[i][4] < 17:
            little += 1
        elif a[i][4] > 24:
            excessive += 1
        else:
            normal += 1
    # BMI
    if (a[i][1] / ((a[i][2] / 100) ** 2)) < 18.5:
        little += 1
    elif (a[i][1] / ((a[i][2] / 100) ** 2)) > 24:
        excessive += 1
    else:
        normal += 1

    # obesity
    if a[i][5] < -10:
        little += 1
    elif a[i][5] > 10:
        excessive += 1
    else:
        normal += 1

    if little > excessive and little > normal:
        a[i][6] = 0  # Malnutrition
    if excessive > little and excessive > normal:
        a[i][6] = 1  # Overnutrition
    if normal > little and normal > excessive:
        a[i][6] = 2  # Health

df = pd.DataFrame(a.T, index=data)
df = df.T

#split train & test
x_train, x_test, y_train, y_test = train_test_split(df[["sex","weight","hieght","visceral_fat","body_fat","obesity"]], df[["status"]], test_size=0.33, random_state=None)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = None) #max_depth, min_samples_leaf
clf_entropy.fit(x_train, y_train)

y_predict = clf_entropy.predict(x_test)

accuracy_score(y_test, y_predict)