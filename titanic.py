import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
import pandas as pd
import time
#train

start = time.time()

train = pd.read_csv("train.csv")

train["Sex"] = train["Sex"][:].replace("male",1)
train["Sex"] = train["Sex"][:].replace("female",0)

train["Embarked"] = train["Embarked"][:].replace("S",2)
train["Embarked"] = train["Embarked"][:].replace("Q",1)
train["Embarked"] = train["Embarked"][:].replace("C",0)

train = train.drop("Cabin",axis = 1)
train = train.drop("Name",axis = 1)
train = train.drop("Ticket",axis = 1)
train = train.drop("PassengerId",axis = 1)
train = train.fillna(train["Fare"].mean())
train = train.fillna(train["Age"].mean())

train_X = np.array(train.drop("Survived",axis = 1))
train_y = np.array(train["Survived"])


#test
test = pd.read_csv("test.csv")

df = test[:]
PassengerId = test["PassengerId"][:]

test["Sex"] = test["Sex"][:].replace("male",1)
test["Sex"] = test["Sex"][:].replace("female",0)

test["Embarked"] = test["Embarked"][:].replace("S",2)
test["Embarked"] = test["Embarked"][:].replace("Q",1)
test["Embarked"] = test["Embarked"][:].replace("C",0)

test = test.drop("Cabin",axis = 1)
test = test.drop("Name",axis = 1)
test = test.drop("Ticket",axis = 1)
test = test.drop("PassengerId",axis=1)
test = test.fillna(test["Fare"].mean())
test = test.fillna(test["Age"].mean())

test_X = test[:]

model = Sequential()

model.add(Dense(512,input_dim = 7,activation = "relu"))
model.add(Dense(256,activation = "relu"))
model.add(Dense(128,activation = "relu"))
model.add(Dense(64,activation = "sigmoid"))
model.add(Dense(32,activation = "relu"))
model.add(Dense(16,activation = "sigmoid"))
model.add(Dense(8,activation = "relu"))
model.add(Dense(4,activation = "sigmoid"))
model.add(Dense(2,activation = "relu"))
model.add(Dense(1,activation = "relu"))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_X,train_y, nb_epoch=150, batch_size=20)


scores = model.evaluate(train_X, train_y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
predict = model.predict(test_X)
predictions = np.round(np.array(predict))
predictions = np.ravel(predictions)



#print(predictions)

#StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,'Survived': predictions })
#StackingSubmission.to_csv("StackingSubmission.csv", index=False)

df["Survived"] = predictions

df[["PassengerId","Survived"]].to_csv("submission.csv",index=False)

end_time = time.time() - start
print(end_time)
