import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)

#X = input[:-81]
#y = output[:-81]

#X_test = input[-81:]
#y_test = output[-81:]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

n_folds = 3

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_val_score(lasso, X_train, y_train, cv=n_folds, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))	

scores, scores_std = np.array(scores), np.array(scores_std)

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

lasso_cv = LassoCV(alphas=alphas, random_state=0)
k_fold = KFold(3)

print("Answer to the bonus question:",
      "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold.split(X_train, y_train)):
    lasso_cv.fit(X_train[train], y_train[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X_train[test], y_train[test])))
    #if()
    print('Coefficients: \n', lasso_cv.coef_)
    print("Mean squared error: %.2f"
      % np.mean((lasso_cv.predict(X_test) - y_test) ** 2))
    print('Alpha: %.5f' % lasso_cv.alpha_)
    print('Variance score: %.2f' % lasso_cv.score(X_test, y_test))

print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially.")

plt.show()

