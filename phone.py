
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sns.set()
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import pickle 

data1 = pd.read_csv("phon.csv")

data1=data1.drop(['clock_speed','m_dep','n_cores','mobile_wt','fc','four_g','sc_h','sc_w','touch_screen','px_width','dual_sim','three_g','talk_time','pc','wifi','blue'],axis=1)

x = data1.iloc[:, :-1].values
y = data1.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l1','l2', 'elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x,y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

hlreg = LogisticRegression(C=100,penalty='l2',solver='newton-cg')
hlreg.fit(x_train, y_train)
y_pred = hlreg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy of the Logistic Regression Model: ",accuracy)


filename='PhonePrice_pred'
pickle.dump(hlreg,open(filename,'wb'))