import pandas as pd
import numpy as np

#este arquivo refere-se ao tratamento de valores nulos e outliers, porém, 
#algoritmos de árvores não são afetados pelos outliers, logo, 
#fica a nosso cargo deixa-los ou não

#inputando o Item_Weight pela média 
#determinando a média de peso por cada item
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

#obtendo uma variável booleana para mostrar quando é nulo
miss_bool = data['Item_Weight'].isnull()
print('Original #missing: %d' %sum(miss_bool))
data.loc[miss_bool,'Item_Weight']  = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.at[x,'Item_Weight'])
print('Final #missing: %d' % sum(data['Item_Weight'].isnull()))
      
#inputando Outlet_Size com moda
from scipy.stats import mode

#determinando o mode para cada
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: mode(x).mode[0]))
print('Mode para cada Outlet_Type:')
print(outlet_size_mode)

#obtendo a variável booleana
miss_bool = data['Outlet_Size'].isnull()

#inputando
print('\nOriginal #missing: %d' %sum(miss_bool))
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print('Final #missing: %d' %sum(data['Outlet_Size'].isnull()))
      
#FEATURE ENGINEERING
#modificando Item_Visibility
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')  
miss_bool = data['Item_Visibility'] == 0
print('Number of 0 values initially: %d' %sum(miss_bool))
data.loc[miss_bool, 'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

#verificando a comparação de visibilidade de um produto em uma determinada loja com a média das demais
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)
print(data['Item_Visibility_MeanRatio'].describe())

#criando uma nova coluna de Item_Identifier separando os FD, NC e DR
#obtendo as duas primeiras letras de cada identificador
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

#renomeando as linhas para melhorar a intuição
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({ 'FD': 'Food',
                                                            'NC': 'Non-Consumable',
                                                            'DR': 'Drinks'})

data['Item_Type_Combined'].value_counts()


#determinando os anos de operação de cada estabelecimento
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

#modificando a categoria Item_Fat_Content, ou seja, normalizar as 2 categorias 
#normalizando a categoria low fat
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({ 'LF': 'Low Fat',
                                                            'reg': 'Regular',
                                                            'low fat': 'Low Fat'})

print(data['Item_Fat_Content'].value_counts())

import pandas as pd

#fazendo com que os não-consumíveis sejam uma categoria diferente
data.loc[data['Item_Type_Combined'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
data['Item_Fat_Content'].value_counts()

#codificando as variáveis categóricas
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


#codificando o One Hot
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])


#exportando os dados
#retirando as colunas que foram convertidas em diferentes tipos
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)   

#separando novamente em train e test
train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']

#retirando colunas descnecessárias
test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)
train.drop(['source'], axis=1, inplace=True)
 
#exportando suas respectivas versões
train.to_csv('train_modified.csv', index=False)
test.to_csv('test_modified.csv', index=False)










































  