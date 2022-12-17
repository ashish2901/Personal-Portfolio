import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

df = pd.read_csv('E:/DATA SCIENCE/Big data/Project 2/health care diabetes.csv')
print(df.head())
print(df.info())
print(df.describe())
print("Standard Deviation of each variables are ==> ",df.apply(np.std))

print('datatypes count-->',df.dtypes.value_counts())


plt.figure(figsize=(8,4),dpi=100)
plt.xlabel('Glucose Class')
plt.title('Glucose Class')
df['Glucose'].plot.hist()
sns.set_style(style='darkgrid')
print("Mean of Glucose level is :-", df['Glucose'].mean())
print("Datatype of Glucose Variable is:",df['Glucose'].dtypes)
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())

plt.figure(figsize=(8,4),dpi=100)
plt.xlabel('BloodPressure Class')
plt.title('BloodPressure Class')
df['BloodPressure'].plot.hist()

sns.set_style(style='darkgrid')
print("Mean of BloodPressure level is :-", df['BloodPressure'].mean())
print("Datatype of BloodPressure Variable is:",df['BloodPressure'].dtypes)
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())

plt.figure(figsize=(8,4),dpi=100)
plt.xlabel('SkinThickness Class')
plt.title('SkinThickness Class')
df['SkinThickness'].plot.hist()
sns.set_style(style='darkgrid')
print("Mean of SkinThickness is :-", df['SkinThickness'].mean())
print("Datatype of SkinThickness Variable is:",df['SkinThickness'].dtypes)
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())

plt.figure(figsize=(6,4),dpi=100)
plt.xlabel('Insulin Class')
plt.title('Insulin Class')
df['Insulin'].plot.hist()
sns.set_style(style='darkgrid')
print("Mean of Insulin is :-", df['Insulin'].mean())
print("Datatype of Insulin Variable is:",df['Insulin'].dtypes)
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())

plt.figure(figsize=(6,4),dpi=100)
plt.xlabel('BMI Class')
plt.title('BMI Class')
df['BMI'].plot.hist()
sns.set_style(style='darkgrid')
print("Mean of BMI is :-", df['BMI'].mean())
print("Datatype of BMI Variable is:",df['BMI'].dtypes)
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())


sns.scatterplot(x= "Glucose" ,y= "BloodPressure",
              hue="Outcome",
              data=df);
plt.show()

sns.scatterplot(x= "BMI" ,y= "Insulin",
              hue="Outcome",
              data=df);

plt.show()
sns.scatterplot(x= "SkinThickness" ,y= "Insulin",
              hue="Outcome",
              data=df);
plt.show()

#correlation matrix
print('Correlation of features-->',df.corr())
#create correlation heat map
plt.subplots(figsize=(8,8))
sns.heatmap(df.corr(),annot=True,cmap='viridis')  ### gives correlation value
plt.show()

#Logistic Regreation and model building

features = df.iloc[:,[0,1,2,3,4,5,6,7]].values
label = df.iloc[:,8].values

#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=0.2,random_state =10)
#Create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
print('logistic regression xtrain,ytrain score-->',model.score(X_train,y_train))
print('logistic regression xtest,ytest score-->',model.score(X_test,y_test))

from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(label,model.predict(features))
print('Confusion matrix logistic regression-->',cm_LR)

from sklearn.metrics import classification_report
print('logistic regression classification report-->',classification_report(label,model.predict(features)))

#K-NN
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=7,metric='minkowski',p = 2)
model2.fit(X_train,y_train)
print('KNN xtrain and ytrain score-->',model2.score(X_train,y_train))
print('KNN xtest and ytest score-->',model2.score(X_test,y_test))
cm_KNN= confusion_matrix(label,model2.predict(features))
print('Confusion matrix KNN-->',cm_KNN)
print('KNN classification report-->',classification_report(label,model2.predict(features)))

#Decission Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(max_depth=5)
model3.fit(X_train,y_train)
print('Decision tree xtrain ytrain score-->',model3.score(X_train,y_train))
print('Decision tree xtest ytest score-->',model3.score(X_test,y_test))
cm_DT= confusion_matrix(label,model3.predict(features))
print('Confusion matrix Decision tree-->',cm_DT)
print('Decision Tree classification report-->',classification_report(label,model3.predict(features)))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=11)
model4.fit(X_train,y_train)
print('Random forest xtrain and ytrain score-->',model4.score(X_train,y_train))
print('Random forest xtest and ytest score-->',model4.score(X_test,y_test))
cm_RF= confusion_matrix(label,model4.predict(features))
print('Confusion matrix Random forest-->',cm_RF)
print('Random Forest classification report-->',classification_report(label,model4.predict(features)))

#Precision Recall Curve for Logistic Regression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('Logistic Regression--> f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.',color='r',label='Logistic regression')
plt.legend()
#Precision Recall Curve for KNN
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model2.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model2.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('KNN--> f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.',color='y',label='KNN')
plt.legend()
#Precision Recall Curve for Decission Tree Classifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

from sklearn.metrics import average_precision_score
# predict probabilities
probs = model3.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model3.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('Decision Tree--> f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.',color='g',label='Decision Tree')
plt.legend()
#Precision Recall Curve for Random Forest
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model4.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict clasmetricss values
yhat = model4.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('Random forest--> f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.',color='m',label='Random Forest')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision recall curve for LR,KNN,DT and Random forest')
plt.show()




#ROC COMPARISION
#Preparing ROC Curve for logistic regression (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# predict probabilities
probs = model.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
TP = cm_LR[1][1]
TN = cm_LR[0][0]
FP = cm_LR[0][1]
FN = cm_LR[1][0]
sensitivity_LR = (TP / float(TP + FN))
specificity_LR = (TN / float(TN + FP))
print('sensitivity_LR-->',sensitivity_LR)
print('sensitivity_LR-->',specificity_LR)
print('AUC logistic regression: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',color='r',label='logistic regression')

#Preparing ROC Curve KNN (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# predict probabilities
probs = model2.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
TP = cm_KNN[1][1]
TN = cm_KNN[0][0]
FP = cm_KNN[0][1]
FN = cm_KNN[1][0]
sensitivity_KNN = (TP / float(TP + FN))
specificity_KNN = (TN / float(TN + FP))
print('sensitivity_KNN-->',sensitivity_KNN)
print('specificity_KNN-->',specificity_KNN)
print('AUC KNN: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',color='y',label='KNN')

# plt.show()
#Preparing ROC Curve Random forest (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# predict probabilities
probs = model4.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
TP = cm_RF[1][1]
TN = cm_RF[0][0]
FP = cm_RF[0][1]
FN = cm_RF[1][0]
sensitivity_RF = (TP / float(TP + FN))
specificity_RF = (TN / float(TN + FP))
print('sensitivity_RF-->',sensitivity_RF)
print('specificity_RF-->',specificity_RF)
print('AUC Random forest: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='Random forest',color='g')
plt.legend()

#ROC Curve Decision tree (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# predict probabilities
probs = model3.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
TP = cm_DT[1][1]
TN = cm_DT[0][0]
FP = cm_DT[0][1]
FN = cm_DT[1][0]
sensitivity_DT = (TP / float(TP + FN))
specificity_DT = (TN / float(TN + FP))
print('sensitivity DT-->',sensitivity_DT)
print('specificity_DT-->',specificity_DT)
print('AUC Decision Tree: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='Decision tree',color='m')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC for Logistic regression,KNN,Random forest,Decision tree')
plt.show()

#Support Vector Classifier
from sklearn.svm import SVC
model5 = SVC(kernel='rbf',gamma='auto')
model5.fit(X_train,y_train)
print('SVM xtrain and ytrain score-->',model5.score(X_train,y_train))
print('SVM xtest and ytest score-->',model5.score(X_test,y_test))
cm_SVM= confusion_matrix(label,model5.predict(features))
print('Confusion matrix logistic regression-->',cm_SVM)
print('SVM classification report-->',classification_report(label,model5.predict(features)))






