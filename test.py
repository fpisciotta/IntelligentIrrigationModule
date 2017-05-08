import matplotlib.pyplot as plt
import csv
import numpy as np
import requests
import json

from regression import generateModel
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import chi2
url = 'https://sjsusmartfarm-backend.herokuapp.com/water-consumption-prediction/machine-learning'
#url = 'http://localhost:3000/weather_history/machine_learning'
ifile  = open('dataset.csv', "r")
read = list(csv.reader(ifile, delimiter=','))
read.pop(0)
input,output=[],[]
for row in read:
    #input.append(row[4:6]+row[7:9]+row[11:13])
    input.append(row[3:4]+row[6:15])
    #input.append(row[2:13]);
    output.append(row[15]);
print('Input',input[0])
print('Output',output[0])
input=np.array(input).astype(np.float)
output=np.array(output).astype(np.float)
input=np.array(input)
output=np.array(output)

max = -1;


X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)
##Preprocessing data
scaler = preprocessing.MinMaxScaler()
#scaler = preprocessing.MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test)
scale = scaler.scale_
###Runing different linear regression algorithms######
#scores, pvalues = chi2(X_train, y_train);
algorithmsToTest = ['linear','ridge','ridgeCV', 'lasso', 'bayesianRidge','SGDRegressor','MLPRegressor']
results = [];
for val in algorithmsToTest:
    print('Algorithm: ',val);
    
    regr = generateModel(X_train,y_train, val)        
    # The coefficients
    if val != 'MLPRegressor':
        print('Coefficients: \n', regr.coef_)
    #else:
        #coefs_mlp = regr.coefs_[0][regr.n_layers_-1]
        #print('Coefficients: \n', (regr.coefs_[len(regr.coefs_)-1]).T)   
    # The coefficients
    #print('Residues: \n', regr.residues_)
    # The mean squared error
    print("Mean squared error: %.10f"
          % np.mean((regr.predict(X_test) - y_test) ** 2))
    variance_score = regr.score(X_test, y_test)
    print('Variance score: %.10f' % variance_score)
    results.append(variance_score);
    if variance_score > max :
        max = variance_score;
        if val != 'MLPRegressor':
            coef_max = regr.coef_;
        model = regr;


#print("Mean squared error: %.10f"
#          % np.mean((predicted - y_test) ** 2))
print('Max variance score:', max);
print('Max coefficients: ',coef_max);

#print('P-values: ',pvalues);                 
    
##Make post request to back-end to store machine learning algorithms coeffs.
#payload = {}
#payload['coeffs'] = ','.join(str(x) for x in coef_max) # '0,3,5'
#payload['variance_score'] = str(max)
#payload['features_scale'] = ','.join(str(x) for x in scale)   
#json = json.dumps(payload)
#print("Json: ", json)
## POST with form-encoded data
#headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
#r = requests.post(url, data=json,headers=headers)     
#print("Response: ", r.text)
#print("Status code: ",r.status_code)
# Plot outputs
#predicted = model.predict(X_test);
#fig, ax = plt.subplots()
#s = [20*4**n for n in range(len(X_test))]
#ax.scatter(y_test, predicted, s=s, c='r')
#ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()

plt.scatter(model.predict(X_train), model.predict(X_train)-y_train, c='b',s=10,alpha=0.5)
plt.scatter(model.predict(X_test),model.predict(X_test)-y_test,c='g',s=10)
plt.hlines(y = 0, xmin = -1, xmax = 1)
plt.ylabel('Residuals')
labels = ['','LR','Ridge','RCV', 'Lasso', 'BR','SGDR','MLPR']
fig = plt.figure()
fig.suptitle('Algorithms Comparison')
ax = fig.add_subplot(111)
width = 1/1.5
plt.bar(range(len(results)), results, width, color="blue")
ax.set_xticklabels(labels)
plt.show()

joblib.dump(model, './model/model.pkl') 
