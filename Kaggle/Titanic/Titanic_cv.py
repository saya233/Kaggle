'''6. cv交叉验证'''

from sklearn import linear_model

from Kaggle.Titanic import Titanic_Second

pd = Titanic_Second.pd
clf= Titanic_Second.clf
df= Titanic_Second.df
'''可以看看现在得到的模型的系数，因为系数和它们最终的判定能力强弱是正相关的
   相关性越接近0的特征可以去掉
'''
pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(clf.coef_.T)})

'''之后靠交叉验证验证这些猜想'''
#看打分情况
from sklearn import  cross_validation
clf=linear_model.LogisticRegression()
all_data=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X=all_data.as_matrix()[:,1:]
y=all_data.as_matrix()[:,0]
print(cross_validation.cross_val_score(clf,X,y,cv=5))

#分割数据
split_train,split_cv=cross_validation.train_test_split(df, test_size=0.3, random_state=0)
tran_df=split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#生成模型
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(tran_df.as_matrix()[:,1:],tran_df.as_matrix()[:,0])

#对cv数据预测
cv_df=split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions=clf.predict(cv_df.as_matrix()[:,1:])
split_cv[predictions != cv_df.as_matrix()[:,0]].drop()

# 去除预测错误的case看原始dataframe数据
#split_cv['PredictResult'] = predictions
origin_data_train = pd.read_csv("Train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
bad_cases



'''7.模型融合'''
from sklearn.ensemble import BaggingRegressor
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()
# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("/Users/MLS/Downloads/logistic_regression_predictions2.csv", index=False)