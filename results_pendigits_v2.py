from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.data import iris_data
from mlxtend.evaluate import paired_ttest_5x2cv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn import linear_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import metrics
import statistics
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict


df1 = pd.read_csv('pendigits.csv', sep=',', header=0)

X_old = df1.iloc[:, :-1].values
y = df1.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X_old)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=123)



clf6 = KNeighborsClassifier(n_neighbors=1)
#score6 = clf6.fit(X_train, y_train).score(X_test, y_test)
#print('KNN accuracy with k = 1: %.2f%%' % (score6*100))

#cv_scores = cross_val_score(clf6, X_train, y_train, cv=5)
#print(cv_scores)


scores = cross_validate(clf6, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
print(scores)
print(scores['test_score'])


np.argmax(scores['test_score'])
#print(scores[0].predict_proba(X_test))

clf6_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]
print(clf6_fitted_model.predict_proba(X_test))
print(clf6_fitted_model.predict(X_test))

expected_y = y_train
#predicted_y = clf6.predict(X_train)

predicted = cross_val_predict(clf6, X_train, y_train, cv=5)
print(predicted)

print("KNN accuracy with k = 1:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])

clf6_precision=[]
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision']))
clf6_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision']))


clf7 = KNeighborsClassifier(n_neighbors=3)
#score7 = clf7.fit(X_train, y_train).score(X_test, y_test)
#print('KNN accuracy with k = 3: %.2f%%' % (score7*100))

scores = cross_validate(clf7, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf7_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]


expected_y = y_train
#predicted_y = clf7.predict(X_train)

predicted = cross_val_predict(clf7, X_train, y_train, cv=5)
print(predicted)

print("KNN accuracy with k = 3 :",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])

clf7_precision=[]
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision']))
clf7_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision']))

clf8 = KNeighborsClassifier(n_neighbors=5)
#score8 = clf8.fit(X_train, y_train).score(X_test, y_test)
#print('KNN accuracy with k = 5: %.2f%%' % (score8*100))

scores = cross_validate(clf8, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf8_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf8.predict(X_train)

predicted = cross_val_predict(clf8, X_train, y_train, cv=5)
print(predicted)

print("KNN accuracy with k = 5:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))


print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])

clf8_precision=[]
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision']))
clf8_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision']))

clf19 = linear_model.Perceptron(tol=1e-3, random_state=1)
#score19 = clf19.fit(X_train, y_train).score(X_test, y_test)
#print('Linear Perceptron accuracy: %.2f%%' % (score19*100))

scores = cross_validate(clf19, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf19_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf19.predict(X_train)

predicted = cross_val_predict(clf19, X_train, y_train, cv=5)
print(predicted)

print("Linear Perceptron accuracy: ",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])



clf10 = MLPClassifier(solver='sgd', hidden_layer_sizes=(64,), random_state=1, max_iter=1000)
#score10 = clf10.fit(X_train, y_train).score(X_test, y_test)
#print('MLP with hidden layer size 64: %.2f%%' % (score10*100))

scores = cross_validate(clf10, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf10_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf10.predict(X_train)

predicted = cross_val_predict(clf10, X_train, y_train, cv=5)
print(predicted)

print("MLP with hidden layer size 64:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])


clf9 = MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), random_state=1, max_iter=1000)
#score9 = clf9.fit(X_train, y_train).score(X_test, y_test)
#print('MLP with hidden layer size 10: %.2f%%' % (score9*100))

scores = cross_validate(clf9, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf9_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf9.predict(X_train)

predicted = cross_val_predict(clf9, X_train, y_train, cv=5)
print(predicted)

print("MLP with hidden layer size 10:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])



clf13 = MLPClassifier(solver='sgd', hidden_layer_sizes=(37,), random_state=1, max_iter=1000)
#score13 = clf13.fit(X_train, y_train).score(X_test, y_test)
#print('MLP with hidden layer size 37 (D+K)/2 : %.2f%%' % (score13*100))

scores = cross_validate(clf13, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf13_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf13.predict(X_train)

predicted = cross_val_predict(clf13, X_train, y_train, cv=5)
print(predicted)

print("MLP with hidden layer size 37 (D+K)/2 :",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])



clf14 = MLPClassifier(solver='sgd', hidden_layer_sizes=(74,), random_state=1, max_iter=1000)
#score14 = clf14.fit(X_train, y_train).score(X_test, y_test)
#print('MLP with hidden layer size 74 (D+K) : %.2f%%' % (score14*100))

scores = cross_validate(clf14, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf14_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf14.predict(X_train)

predicted = cross_val_predict(clf14, X_train, y_train, cv=5)
print(predicted)

print("MLP with hidden layer size 74 (D+K) :",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])


clf15 = MLPClassifier(solver='sgd', hidden_layer_sizes=(148,), random_state=1, max_iter=1000)
#score15 = clf15.fit(X_train, y_train).score(X_test, y_test)
#print('MLP with hidden layer size 148 (D+K)*2 : %.2f%%' % (score15*100))

scores = cross_validate(clf15, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf15_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf15.predict(X_train)

predicted = cross_val_predict(clf15, X_train, y_train, cv=5)
print(predicted)

print("MLP with hidden layer size 148 (D+K)*2 :",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])

clf15_precision=[]
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision']))
clf15_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision']))

clf11 = svm.SVC(kernel='linear', probability=True)
#score11 = clf11.fit(X_train, y_train).score(X_test, y_test)
#print('SVM with linear kernel accuracy is : %.2f%%' % (score11*100))

scores = cross_validate(clf11, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf11_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf11.predict(X_train)

predicted = cross_val_predict(clf11, X_train, y_train, cv=5)
print(predicted)

print("SVM with linear kernel accuracy is :",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])

clf11_precision=[]
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision']))
clf11_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision']))

clf4 = svm.SVC(kernel='poly', degree=2, probability=True)
#score4 = clf4.fit(X_train, y_train).score(X_test, y_test)
#print('SVM with polynomial kernel degree 2 accuracy is : %.2f%%' % (score4*100))

scores = cross_validate(clf4, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf4_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf4.predict(X_train)

predicted = cross_val_predict(clf4, X_train, y_train, cv=5)
print(predicted)

print("SVM with polynomial kernel degree 2 accuracy is : ",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])

clf4_precision=[]
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision']))
clf4_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision']))


clf12 = svm.SVC(kernel='rbf', probability=True)
#score12 = clf12.fit(X_train, y_train).score(X_test, y_test)
#print('SVM with radial kernel: %.2f%%' % (score12*100))

scores = cross_validate(clf12, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf12_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf12.predict(X_train)

predicted = cross_val_predict(clf12, X_train, y_train, cv=5)
print(predicted)

print("SVM with radial kernel:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])

clf12_precision=[]
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision']))
clf12_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision']))



clf2 = DecisionTreeClassifier()
#score2 = clf2.fit(X_train, y_train).score(X_test, y_test)
#print('Decision tree accuracy: %.2f%%' % (score2*100))

scores = cross_validate(clf2, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf2_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf2.predict(X_train)

predicted = cross_val_predict(clf2, X_train, y_train, cv=5)
print(predicted)

print("Decision tree accuracy:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])


clf16 = GaussianNB()
#score16 = clf16.fit(X_train, y_train).score(X_test, y_test)
#print('Naive Bayes accuracy: %.2f%%' % (score16*100))

scores = cross_validate(clf16, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf16_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf16.predict(X_train)

predicted = cross_val_predict(clf16, X_train, y_train, cv=5)
print(predicted)

print("Naive Bayes accuracy:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])



clf17 = LDA()
#score17 = clf17.fit(X_train, y_train).score(X_test, y_test)
#print('LDA accuracy: %.2f%%' % (score17*100))

scores = cross_validate(clf17, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf17_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf17.predict(X_train)

predicted = cross_val_predict(clf17, X_train, y_train, cv=5)
print(predicted)

print("LDA accuracy:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])



clf18 = QDA()
#score18 = clf18.fit(X_train, y_train).score(X_test, y_test)
#print('QDA accuracy: %.2f%%' % (score18*100))

scores = cross_validate(clf18, X_train, y_train, cv=5, scoring='accuracy', return_estimator=True)
clf18_fitted_model = scores['estimator'][np.argmax(scores['test_score'])]

expected_y = y_train
#predicted_y = clf18.predict(X_train)

predicted = cross_val_predict(clf18, X_train, y_train, cv=5)
print(predicted)

print("QDA accuracy:",metrics.accuracy_score(expected_y, predicted))

#print(metrics.classification_report(expected_y, predicted))
#print(metrics.classification_report(expected_y, predicted, output_dict = True))
#print(metrics.confusion_matrix(expected_y, predicted_y))

print(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision'])
print(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision'])

clf18_precision=[]
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['0']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['1']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['2']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['3']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['4']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['5']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['6']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['7']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['8']['precision']))
clf18_precision.append(float(metrics.classification_report(expected_y, predicted, output_dict = True)['9']['precision']))

def union_accuracy_median(a,b, c, X_test, y_test):
    unionnn = []
    for i in range(len(X_test)):
        unionnn.append([])
        for j in range(len(a[i])):

            tmp = []
            tmp.append(float(a[i][j]))
            tmp.append(float(b[i][j]))
            tmp.append(float(c[i][j]))


            tmp_median = statistics.median(tmp)
            unionnn[i].append(tmp_median)

    wrong = []
    for i in range(len(unionnn)):
        maxx = np.argmax(unionnn[i])
        if maxx != y_test[i]:
            wrong.append(i)

    t_1 = len(wrong)
    t_2 = len(y_test)

    print(("\n"))
    print("Final union accuracy for MEDIAN is {:.2f}".format((t_2 - t_1) / t_2 * 100))

def union_accuracy_average(a,b, c, X_test, y_test):
    unionnn = []
    for i in range(len(X_test)):
        unionnn.append([])
        for j in range(len(a[i])):

            tmp = []
            tmp.append(float(a[i][j]))
            tmp.append(float(b[i][j]))
            tmp.append(float(c[i][j]))
            tmp_mean = statistics.mean(tmp)
            unionnn[i].append(tmp_mean)

    wrong = []
    for i in range(len(unionnn)):
        maxx = np.argmax(unionnn[i])
        if maxx != y_test[i]:
            wrong.append(i)

    t_1 = len(wrong)
    t_2 = len(y_test)

    print(("\n"))
    print("Final union accuracy for AVG MEAN is {:.2f}".format((t_2 - t_1) / t_2 * 100))

def union_accuracy_max(a,b, c, X_test, y_test):
    unionnn = []
    for i in range(len(X_test)):
        unionnn.append([])
        for j in range(len(a[i])):

            unionnn[i].append( max(float(a[i][j]),float(b[i][j]),float(c[i][j])  ))


    wrong = []
    for i in range(len(unionnn)):
        maxx = np.argmax(unionnn[i])
        if maxx != y_test[i]:
            wrong.append(i)

    t_1 = len(wrong)
    t_2 = len(y_test)

    print(("\n"))
    print("Final union for Max is {:.2f}".format((t_2 - t_1) / t_2 * 100))

def union_accuracy_multiply(a,b,c,X_test, y_test):
    unionnn = []
    for i in range(len(X_test)):
        unionnn.append([])
        for j in range(len(a[i])):

            #unionnn[i].append( max(float(a[i][j]),float(b[i][j]),float(c[i][j]) ) )
            unionnn[i].append(float(a[i][j]) * float(b[i][j]) * float(c[i][j]) )

    wrong = []
    for i in range(len(unionnn)):
        maxx = np.argmax(unionnn[i])
        if maxx != y_test[i]:
            wrong.append(i)

    t_1 = len(wrong)
    t_2 = len(y_test)

    print(("\n"))
    print("Final union for Multiply is {:.2f}".format((t_2 - t_1) / t_2 * 100))

def union_accuracy_weighted_median(a,b, c, X_test, y_test, f, g, h):
    unionnn = []
    for i in range(len(X_test)):
        unionnn.append([])
        for j in range(len(a[i])):
            tmp = []
            tmp.append(float(a[i][j]) * f[j])
            tmp.append(float(b[i][j]) * g[j])
            tmp.append(float(c[i][j]) * h[j])

            tmp_median = statistics.median(tmp)
            unionnn[i].append(tmp_median)

    wrong = []
    for i in range(len(unionnn)):
        maxx = np.argmax(unionnn[i])
        if maxx != y_test[i]:
            wrong.append(i)

    t_1 = len(wrong)
    t_2 = len(y_test)

    print(("\n"))
    print("Final union accuracy for WEIGHTED MEDIAN is  {:.2f}".format((t_2 - t_1) / t_2 * 100))


def union_accuracy_weighted_average(a,b, c, X_test, y_test, f, g, h):
    unionnn = []
    for i in range(len(X_test)):
        unionnn.append([])
        for j in range(len(a[i])):
            tmp = []
            tmp.append(float(a[i][j]) * f[j])
            tmp.append(float(b[i][j]) * g[j])
            tmp.append(float(c[i][j]) * h[j])


            tmp_mean = statistics.mean(tmp)
            unionnn[i].append(tmp_mean)

    wrong = []
    for i in range(len(unionnn)):
        maxx = np.argmax(unionnn[i])
        if maxx != y_test[i]:
            wrong.append(i)

    t_1 = len(wrong)
    t_2 = len(y_test)

    print(("\n"))
    print("Final union accuracy for WEIGHTED AVG MEAN is  {:.2f}".format((t_2 - t_1) / t_2 * 100))



union_accuracy_median(clf6_fitted_model.predict_proba(X_test), clf12_fitted_model.predict_proba(X_test), clf7_fitted_model.predict_proba(X_test), X_test, y_test)

union_accuracy_average(clf6_fitted_model.predict_proba(X_test), clf12_fitted_model.predict_proba(X_test), clf7_fitted_model.predict_proba(X_test), X_test, y_test)

union_accuracy_max(clf6_fitted_model.predict_proba(X_test), clf12_fitted_model.predict_proba(X_test), clf7_fitted_model.predict_proba(X_test), X_test, y_test)

union_accuracy_multiply(clf6_fitted_model.predict_proba(X_test), clf12_fitted_model.predict_proba(X_test), clf7_fitted_model.predict_proba(X_test), X_test, y_test)

union_accuracy_weighted_median(clf6_fitted_model.predict_proba(X_test), clf12_fitted_model.predict_proba(X_test), clf7_fitted_model.predict_proba(X_test),  X_test, y_test, clf6_precision, clf12_precision, clf7_precision)

union_accuracy_weighted_average(clf6_fitted_model.predict_proba(X_test), clf12_fitted_model.predict_proba(X_test), clf7_fitted_model.predict_proba(X_test),  X_test, y_test, clf6_precision, clf12_precision, clf7_precision)
