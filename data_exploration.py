import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#juntando os datasets de treinamento e teste para realizar o tratamento dos dados
#criando uma nova coluna para mostrar quais dados são de treinamento e teste
train['source'] = 'train'
test['source'] = 'test'

data = pd.concat([train, test], ignore_index=True)
print(train.shape, test.shape, data.shape)

#verificando se há valores nulos
data.apply(lambda x: sum(x.isnull()))

#olhando algumas estatísticas básicas
data.describe()

#verificando o número de valores únicos em cada variável categórica
data.apply(lambda x: len(x.unique()))

#filtrando os valores únicos e também excluindo os ID's e source
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x] == 'object']

categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier',
                        'Outlet_Identifier', 'source']]

#mostrando a frequência das categorias
for col in categorical_columns:
    print('\nFrequência das categorias para a variável %s'%col)
    print(data[col].value_counts())
