import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from numpy import genfromtxt
from sklearn.model_selection import train_test_split



def mySVM(XTest,XTrain,YTest,YTrain):
	clf=SVC(probability=True,kernel='linear')
	clf.fit(XTrain,YTrain)
	YTrainPred=clf.predict(XTrain)
	YPred=clf.predict(XTest)

	YProba=clf.predict_proba(XTest)
	print("Confusion matrix of SVM")
	print(confusion_matrix(YPred,YTest))
	cv_10scores = cross_val_score(estimator=clf,X=XTrain,y=YTrain,cv=10)
	#Calculate Accuracies
	print ("SVM Training Accuracy Score : ",accuracy_score(YTrain,YTrainPred))
	print ("SVM Testing Accuracy Score : ",accuracy_score(YTest,YPred))
	print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_10scores.mean(), cv_10scores.std() * 2))
	skplt.metrics.plot_roc_curve(YTest, YProba)
	plt.show()



def myNeuralNet2Layer(XTest,XTrain,YTest,YTrain):
	clf=MLPClassifier(hidden_layer_sizes=(50))
	clf.fit(XTrain,YTrain)
	YProba=clf.predict_proba(XTest)
	YTrainPred=clf.predict(XTrain)
	YPred=clf.predict(XTest)
	print("Confusion matrix of 1 Hidden Layer Neural Network")
	print(confusion_matrix(YPred,YTest))


	cv_10scores = cross_val_score(estimator=clf,X=XTrain,y=YTrain,cv=10)
	#Calculate Accuracies
	print ("1 Hidden Layer Neural Network Training Accuracy Score : ",accuracy_score(YTrain,YTrainPred))
	print ("1 Hidden Layer Neural Network Testing Accuracy Score : ",accuracy_score(YTest,YPred))
	print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_10scores.mean(), cv_10scores.std() * 2))

	skplt.metrics.plot_roc_curve(YTest, YProba)
	plt.show()




def myNeuralNet4Layer(XTest,XTrain,YTest,YTrain):
	clf=MLPClassifier(hidden_layer_sizes=(100,50,50))
	clf.fit(XTrain,YTrain)
	YProba=clf.predict_proba(XTest)
	YTrainPred=clf.predict(XTrain)
	YPred=clf.predict(XTest)
	print("Confusion matrix of 3 Hidden Layer Neural Network")
	print(confusion_matrix(YPred,YTest))

	cv_10scores = cross_val_score(estimator=clf,X=XTrain,y=YTrain,cv=10)
	#Calculate Accuracies
	print ("3 Hidden Layer Neural Network Training Accuracy Score : ",accuracy_score(YTrain,YTrainPred))
	print ("3 Hidden Layer Neural Network Testing Accuracy Score : ",accuracy_score(YTest,YPred))
	print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_10scores.mean(), cv_10scores.std() * 2))

	skplt.metrics.plot_roc_curve(YTest, YProba)
	plt.show()

def myLogisticRegression(XTest,XTrain,YTest,YTrain):
	clf=LogisticRegression()
	clf.fit(XTrain,YTrain)
	YProba=clf.predict_proba(XTest)
	YTrainPred=clf.predict(XTrain)
	YPred=clf.predict(XTest)
	print("Confusion matrix of Logistic Regression")
	print(confusion_matrix(YPred,YTest))

	cv_10scores = cross_val_score(estimator=clf,X=XTrain,y=YTrain,cv=10)
	#Calculate Accuracies
	print ("Logistic Regression Training Accuracy Score : ",accuracy_score(YTrain,YTrainPred))
	print ("Logistic Regression Testing Accuracy Score : ",accuracy_score(YTest,YPred))
	print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_10scores.mean(), cv_10scores.std() * 2))

	skplt.metrics.plot_roc_curve(YTest, YProba)
	plt.show()

#non controversial=0 controversial=1
controversialData=np.genfromtxt("Con.csv",delimiter=',')
nonControversialData=np.genfromtxt("Data2.csv",delimiter=',')

nonControversialData=nonControversialData[1:,[1,2,3,4,5,6,7,8]]

controversialData=controversialData[1:,[1,2,3,4,5,6,7,8]]

nonControversialData=np.column_stack((nonControversialData,np.zeros(nonControversialData.shape[0])))

controversialData=np.column_stack((controversialData,np.ones(controversialData.shape[0])))

data=np.concatenate((controversialData,nonControversialData))

np.random.shuffle(data)


trainData,testData=train_test_split(data)


XTrain=trainData[:,[0,1,2,3,4,5,6,7]]
YTrain=trainData[:,[8]]

XTest=testData[:,[0,1,2,3,4,5,6,7]]
YTest=testData[:,[8]]



myLogisticRegression(XTest,XTrain,YTest,YTrain)
mySVM(XTest,XTrain,YTest,YTrain)
myNeuralNet2Layer(XTest,XTrain,YTest,YTrain)
myNeuralNet4Layer(XTest,XTrain,YTest,YTrain)