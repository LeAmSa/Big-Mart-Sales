from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor



train = pd.read_csv('train_modified.csv')
test = pd.read_csv('test_modified.csv')

#definindo a função que receberá todos os modelos de predição

#definindo o alvo
target = 'Item_Outlet_Sales'

IDcol = ['Item_Identifier', 'Outlet_Identifier']

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #adequando o algoritmo com os dados
    alg.fit(dtrain[predictors], dtrain[target])
    
    #atributos do conjunto de treinamento
    dtrain_predictors = alg.predict(dtrain[predictors])
    
    #realizando validação cruzada
    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    print('\nModel Report')
    print('RMSE: %.4g' % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictors)))
    print('CV Score: Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g' % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #prevendo nos dados de teste
    dtest[target] = alg.predict(dtest[predictors])
    
    #exportando arquivos
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    
#modelo de regressão linear    
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')  

#modelo Ridge para lidar com o overfitting do modelo anterior
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05, normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coeficients')

#árvore de decisão
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')
#melhorando o modelo realizando fine tuning e utilizando apenas as 4 primeiras importâncias
preditors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')

#Random Forest
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')










 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    