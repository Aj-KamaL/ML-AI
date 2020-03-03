from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error

def Decision_Tree(X,Y,x_test,y_test):
	#fit on train
	clf = DecisionTreeClassifier()
	clf.fit(X,Y)
	
	#predict for train and test
	Y_train_predict=clf.predict(X)
	Y_test_predict=clf.predict(x_test)

	#generate cv10 scores
	cv_10scores = cross_val_score(estimator=clf,X=X,y=Y,cv=10)
	
	#Calculate Accuracies
	print ("Decision Tree Training Accuracy Score : ",accuracy_score(Y,Y_train_predict))
	print ("Decision Tree Testing Accuracy Score : ",accuracy_score(y_test,Y_test_predict))
	print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_10scores.mean(), cv_10scores.std() * 2))
	
	#Print the confusion matrix
	cm=confusion_matrix(y_test,Y_test_predict)
	print ("__Confusion Matrix__")
	print (cm)
	print("")
	train_vs_test(clf,X,Y,x_test,y_test,"Decision_Tree")

def Gaussian_Naive_Bayes(X,Y,x_test,y_test):
	clf = GaussianNB()
	clf.fit(X,Y)
	
	
	Y_train_predict=clf.predict(X)
	Y_test_predict=clf.predict(x_test)	

	#generate cv10 scores
	cv_10scores = cross_val_score(estimator=clf,X=X,y=Y,cv=10)
	
	#Calculate Accuracies
	print ("Gaussian Naive Bayes Training Accuracy Score : ",accuracy_score(Y,Y_train_predict))
	print ("Gaussian Naive Bayes Testing Accuracy Score : ",accuracy_score(y_test,Y_test_predict))
	print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_10scores.mean(), cv_10scores.std() * 2))
	
	#Print the confusion matrix
	cm=confusion_matrix(y_test,Y_test_predict)
	print ("__Confusion Matrix__")
	print (cm)
	print("")

	#Generate ROC curve
	y_predict_probabilities = clf.predict_proba(x_test)[:,1]
	fpr,tpr,thresholds = metrics.roc_curve(y_test,y_predict_probabilities)
	roc_auc = metrics.auc(fpr, tpr)
	plt.figure()
	plt.title("ROC AUC FOR GaussianNB")
	print("ROC AUC ",roc_auc)
	plt.plot(fpr, tpr, color='darkgreen',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()
	train_vs_test(clf,X,Y,x_test,y_test,"GaussianNB")
def Random_Forest(X,Y,x_test,y_test):
	clf = RandomForestClassifier()
	clf.fit(X,Y)
	
	#predict for train and test
	Y_train_predict=clf.predict(X)
	Y_test_predict=clf.predict(x_test)

	#generate cv10 scores
	cv_10scores = cross_val_score(estimator=clf,X=X,y=Y,cv=10)
	
	#Calculate Accuracies
	print ("Random_Forest Training Accuracy Score : ",accuracy_score(Y,Y_train_predict))
	print ("Random_Forest Testing Accuracy Score : ",accuracy_score(y_test,Y_test_predict))
	print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_10scores.mean(), cv_10scores.std() * 2))
	
	#Print the confusion matrix
	cm=confusion_matrix(y_test,Y_test_predict)
	print ("__Confusion Matrix__")
	print (cm)
	print("")
	train_vs_test(clf,X,Y,x_test,y_test,"Random Forest")	

def kneighbours(X,Y,x_test,y_test):
	clf = KNeighborsClassifier(n_neighbors=3)
	clf.fit(X,Y)
	
	#predict for train and test
	Y_train_predict=clf.predict(X)
	Y_test_predict=clf.predict(x_test)

	#generate cv10 scores
	cv_10scores = cross_val_score(estimator=clf,X=X,y=Y,cv=10)
	
	#Calculate Accuracies
	print ("kneighbours Training Accuracy Score : ",accuracy_score(Y,Y_train_predict))
	print ("kneighbours Testing Accuracy Score : ",accuracy_score(y_test,Y_test_predict))
	print("kneighbours Accuracy: %0.2f (+/- %0.2f)" % (cv_10scores.mean(), cv_10scores.std() * 2))
	
	#Print the confusion matrix
	cm=confusion_matrix(y_test,Y_test_predict)
	print ("__Confusion Matrix__")
	print (cm)
	print("")	

	#Generate ROC curve
	y_predict_probabilities = clf.predict_proba(x_test)[:,1]
	fpr,tpr,thresholds = metrics.roc_curve(y_test,y_predict_probabilities)
	roc_auc = metrics.auc(fpr, tpr)
	plt.figure()
	plt.title("ROC AUC FOR KNN")
	print("ROC AUC ",roc_auc)
	plt.plot(fpr, tpr, color='darkgreen',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()
	train_vs_test(clf,X,Y,x_test,y_test,"knn")
def train_vs_test(clf,X,Y,x_test,y_test,name):
	li=np.arange(10,200,2)

	train_errors = []
	test_errors = []
	for i in li:
		clf.fit(X[0:i],Y[0:i])

		y_pred=clf.predict(X[0:i])
		mse1=mean_squared_error(Y[0:i], y_pred)
		train_errors.append(mse1)
		y_pred=clf.predict(x_test)
		mse2=mean_squared_error(y_test,y_pred)
		test_errors.append(mse2)

	plt.title("Train vs test for "+name)
	plt.plot(li,train_errors, color='darkred',lw=2,label="Train Errors")
	plt.plot(li,test_errors, color='darkgreen',lw=2,label="Test Errors")
	plt.legend()
	plt.xlabel('Examples')
	plt.ylabel('Error')
	plt.show()


#along with data cleansing
def read_csv(nameoffile):
	a=pd.read_csv(nameoffile)
	matrix=a.values
	X=matrix[:,1:]
	Y=matrix[:,-1]
	Y=Y.flatten()
	print(X)
	for i in range(0,len(X)):
		for j in range(0,len(X[0])):
			X[i][j]=int(X[i][j])
	print(X)


	return X,Y


#replace name.csv with nameoffile.csv
X,Y=read_csv('Dataset.csv')
X=X.astype('int')
Y=Y.astype('int')

scaler = MinMaxScaler()
scaler.fit(X)
scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

X_train=X_train.astype('int')
X_test=X_test.astype('int')

y_train=y_train.astype('int')
y_test=y_test.astype('int')


kneighbours(X_train,y_train,X_test,y_test)
Random_Forest(X_train,y_train,X_test,y_test)
Decision_Tree(X_train,y_train,X_test,y_test)
Gaussian_Naive_Bayes(X_train,y_train,X_test,y_test)