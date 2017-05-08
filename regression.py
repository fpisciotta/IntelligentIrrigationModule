
from sklearn import linear_model,neural_network

def generateModel(x_train, y_train, type):
	# Create linear regression object
    
    if type == 'linear' :
        regr = linear_model.LinearRegression()                  
    elif type == "ridge" :
        regr = linear_model.Ridge(alpha = .1)
    elif type == "ridgeCV" :
        regr = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], fit_intercept=False, scoring=None,
    normalize=False)
    elif type == "lasso" :
        regr = linear_model.Lasso(alpha = .1)
    elif type == "bayesianRidge" :
        regr = linear_model.BayesianRidge()
    elif type == "RANSACRegressor" :
        regr = linear_model.RANSACRegressor(random_state=42)
    elif type == "theilSen" :
        regr = linear_model.TheilSenRegressor(random_state=42)
    elif type == "ElasticNet" :
        regr = linear_model.ElasticNet(l1_ratio=0.7)
    elif type == "SGDRegressor" :
        regr = linear_model.SGDRegressor(penalty='elasticnet', alpha=0.001,
                                  l1_ratio=0.25, fit_intercept=False)
    elif type == "MLPRegressor" :
        regr = neural_network.MLPRegressor(hidden_layer_sizes=(200,200,200,200,200,200,200,200,200,200),solver='adam',alpha=0.001, max_iter=10000000)
    
	# Train the model using the training sets
    regr.fit(x_train, y_train)

    return regr