import matplotlib.pyplot as plt
import csv
import numpy as np
import requests
import json
from linearRegression import generateModel
from sklearn.cross_validation import train_test_split
url = 'https://sjsusmartfarm-backend.herokuapp.com/weather_history/machine_learning'
#url = 'http://localhost:3000/weather_history/machine_learning'
ifile  = open('dataset.csv', "r")
read = list(csv.reader(ifile, delimiter=','))
read.pop(0)
input,output=[],[]
for row in read:
    #input.append(row[4:6]+row[7:9]+row[11:13])
    input.append(row[6:15])
    output.append(row[16])
print('Input',input[0])
print('Output',output[0])
input=np.array(input).astype(np.float)
output=np.array(output).astype(np.float)
input=np.array(input)
output=np.array(output)

max = -1;


X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)

regr = generateModel(X_train,y_train, "linear")
print('Linear Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The coefficients
#print('Residues: \n', regr.residues_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
variance_score = regr.score(X_test, y_test)
print('Variance score: %.2f' % variance_score)

if variance_score > max :
    max = variance_score;
    coef_max = regr.coef_;

regr = generateModel(X_train,y_train, "ridge")

print('Ridge Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
variance_score = regr.score(X_test, y_test)
print('Variance score: %.2f' % variance_score)

if variance_score > max :
    max = variance_score;
    coef_max = regr.coef_;

regr = generateModel(X_train,y_train, "lasso")
print('Lasso Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
variance_score = regr.score(X_test, y_test)
print('Variance score: %.2f' % variance_score)

if variance_score > max :
    max = variance_score;
    coef_max = regr.coef_;
    
regr = generateModel(X_train,y_train, "theilSen")
print('TheilSen Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
variance_score = regr.score(X_test, y_test)
print('Variance score: %.2f' % variance_score)

if variance_score > max :
    max = variance_score;
    coef_max = regr.coef_;

regr = generateModel(X_train,y_train, "ElasticNet")
print('ElasticNet Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
variance_score = regr.score(X_test, y_test)
print('Variance score: %.2f' % variance_score)

if variance_score > max :
    max = variance_score;
    coef_max = regr.coef_;

regr = generateModel(X_train,y_train, "bayesianRidge")
print('BayesianRidge Regression \n')
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
variance_score = regr.score(X_test, y_test)
print('Variance score: %.2f' % variance_score)

if variance_score > max :
    max = variance_score;
    coef_max = regr.coef_;

regr = generateModel(X_train,y_train, "RANSACRegressor")
print('RANSACRegressor Regression \n')
# The coefficients
print('Coefficients: \n', regr.estimator_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

print('Size x: %i  and y: %i',X_test.shape,y_test.shape)


print('Max variance score:', max);
print('Max coefficients: ',coef_max);

##Make post request to back-end to store machine learning algorithms coeffs.

payload = {}
payload['coeffs'] = ','.join(str(x) for x in coef_max) # '0,3,5'
payload['variance_score'] = str(max)  
json = json.dumps(payload)
print("Json: ", json)
# POST with form-encoded data
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
r = requests.post(url, data=json,headers=headers)     
print("Response: ", r.text)
print("Status code: ",r.status_code)
# Plot outputs

#plt.scatter(X_test[:,0:3], y_test,  color='black')
#plt.plot(X_test, regr.predict(X_test), color='blue',
#        linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()
