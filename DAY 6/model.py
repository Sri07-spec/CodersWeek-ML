import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('http://iali.in/datasets/Social_Network_Ads.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,4].values

dataset.describe()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


regressor.fit(X, y)


pickle.dump(regressor, open('model1.pkl','wb'))


model = pickle.load(open('model1.pkl','rb'))

from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)
y_pred


pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

df= pd.DataFrame({'Actual':y_test,'Predicted':y_pred})

df

df1=df.head(25)
df1.plot(kind='bar', figsize=(16,10))
plt.grid(which='major' , linestyle='-' , color="orange" )
plt.grid(which='minor' , linestyle=':' , color="black" )
plt.show()


