import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

plt.style.use('seaborn')


# Começamos abrindo ambos os dataframes. 'sales' é o dataset da empresa B e 'comp' é o dataset da competição, 
# constituída das empresas C1, ..., C6.
sales = pd.read_csv('sales.csv')
comp = pd.read_csv('comp_prices.csv')

# Criamos conexões e importamos os csv's para SQL.
conn = sqlite3.connect('data.db')
sales.to_sql('sales', con=conn, if_exists='replace', index=False)
comp.to_sql('comp', con=conn, if_exists='replace', index=False)

# Agregamos os valores por produto e data, somando e tirando as médias das quantidades e rendimentos.
sales_updated = pd.read_sql('SELECT PROD, DATE, CAST(AVG(QTY) AS INT) QTY, AVG(REVENUE) REVENUE FROM \
                                 (SELECT PROD_ID PROD, DATE_ORDER DATE, SUM(QTY_ORDER) QTY, REVENUE FROM sales \
                                  GROUP BY PROD, DATE, REVENUE) \
                             GROUP BY PROD, DATE', conn)

comp_updated = pd.read_sql('SELECT PROD_ID PROD, date(DATE_EXTRACTION) DATE, COMPETITOR, AVG(COMPETITOR_PRICE) REVENUE, PAY_TYPE FROM comp \
                            GROUP BY PROD, DATE', conn)

# Fechamos a conexão após a criação dos dataframes.
conn.close()

# Depois da limpeza, repetimos o processo com os novos datasets para construir um novo database.
conn = sqlite3.connect('new_data.db')
sales_updated.to_sql('sales', con=conn, if_exists='replace', index=False)
comp_updated.to_sql('comp', con=conn, if_exists='replace', index=False)

# Fazemos um join em relação a PROD e DATE.
data = pd.read_sql('SELECT sales.PROD, sales.DATE, sales.QTY, sales.REVENUE REVENUE_B, comp.REVENUE REVENUE_C, comp.PAY_TYPE FROM sales \
                    LEFT JOIN comp ON (sales.PROD = comp.PROD AND sales.DATE = comp.DATE);', conn)

# Este join irá introduzir alguns valores omissos quando a data de sales não tem correspondente em comp. Nesta situação
# a coluna REVENUE_C irá receber o mesmo valor de REVENUE_B.
data.loc[data['REVENUE_C'].isnull(), 'REVENUE_C'] = data.loc[data['REVENUE_C'].isnull(), 'REVENUE_B']

# Os valores omissos de 'PAY_TYPE' serão preenchidos de maneira aleatória.
L= data['PAY_TYPE'].isnull().sum()
values = np.array(np.random.randint(1, 3, size=L), dtype=np.float64)
data.loc[data['PAY_TYPE'].isnull(), 'PAY_TYPE'] = values

# Será de interesse ter uma coluna apenas com os meses.
data['MONTH'] = data['DATE'].str.split('-').str[1].astype('int')

# Normalização dos dados.
data_norm = data.copy()
for col in data_norm.columns:
    if col not in ['PROD', 'DATE', 'MONTH']:
        s = data_norm[col]
        s = ( s - s.mean() )/( s.max() - s.min() )
        data_norm[col] = s
    
# Para cada par (mês, produto) teremos um conjunto de clusters. Começamos agrupando os dados desta maneira.
train_groups = {}
min_month = min(data_norm['MONTH'].unique())
max_month = max(data_norm['MONTH'].unique())
products = sales_updated['PROD'].unique()
for m in range(min_month, max_month+1):
    for p in products:
        train_groups[str(m)+'_'+p] = data_norm[(data_norm['MONTH']==m) & (data_norm['PROD']==p)]
        
# Usamos o método 'elbow' para determinar o número de clusters. 
train_results = {}


def get_results(kmeans, scores):
    """
    Esta função coleta os resultados do método KMeans, de modo que o número de cluster
    deve satisfazer abs(score[i]) < 1e-3, caso contrário assumimos 4 clusters (o máximo 
    possível são 120 clusters).
    """
    
    L = len(scores)
    for i in range(L):
        if abs(scores[i]) < 1e-3:
            return kmeans[i].cluster_centers_
    return kmeans[-1].cluster_centers_

for m in range(min_month, max_month+1):
    for p in products:
        train_dataset = train_groups[str(m)+'_'+p]
        n_cluster = range(1, min(4, train_dataset.shape[0]))
        if train_dataset.size==0:
            pass
        else:
            # Sample é o dataframe relativo ao mês m e produto p.
            sample = train_dataset[train_dataset['PROD']==p][['QTY', 'REVENUE_B', 'REVENUE_C', 'PAY_TYPE']]
            # Aplicação do KMeans sobre sample.
            kmeans = [KMeans(n_clusters=i).fit(sample) for i in n_cluster]
            scores = [kmeans[i].score(sample) for i in range(len(kmeans))]
            # O número de clusters é o primeiro score satisfazendo abs(score[i]) < 1e-3.
            train_results[str(m)+'_'+p] = get_results(kmeans, scores)
    
    
# Para fazer previsões, revertemos a normalização.
def denormalize(df, data):
    """
    Esta função cancela a normalização sobre df, onde a média, mínimo e máximo
    das colunas são obtidos do dataframe original data.
    
    Parameters
    ----------
    df, data: dataframes
        df um dataframe no espaço normalizado de data.
    
    Return
    ------
        Dataframe df sem anormalização.
    """
    
    s = data['QTY']
    df['QTY'] = ( s.max() - s.min() )*df['QTY'] + s.mean()

    for seller in ['B', 'C']:
        s = data['REVENUE_'+seller]
        df['REVENUE_'+seller] = ( s.max() - s.min() )*df['REVENUE_'+seller] + s.mean()
    return pd.DataFrame(df, columns=['QTY', 'REVENUE_B', 'REVENUE_C', 'PAY_TYPE'])


# train_results_updated é a versão não-normalizada de train_results. Se trata de um dicionário com keys no
# formato 'mês_produto', tais que train_results_updated['mês_produto'] é o dataframe dos centróides (obtidos 
# pelo KMeans) em relação ao par (mês, produto).
train_results_updated = {}
for m in range(min_month, max_month+1):
    for p in products:
        n_points = train_groups[str(m)+'_'+p].shape[0]
        if n_points != 0:
            df = pd.DataFrame(train_results[str(m)+'_'+p], columns=['QTY', 'REVENUE_B', 'REVENUE_C', 'PAY_TYPE']).copy()
            train_results_updated[str(m)+'_'+p] = denormalize(df, data) 
            train_results_updated[str(m)+'_'+p]['QTY'] = train_results_updated[str(m)+'_'+p]['QTY'].astype('int').copy()         

def prediction(month, product, revenue_b, revenue_c, pay_type, train_results_updated):
    """
    Dado o mês, o produto e os dois preços, verificamos qual o centróide mais perto da 
    coordenadas dos preços. Este centróide está relacionado a uma certa quantidade que 
    foi obtida durante o treinamento. Esta será a quantidade que usaremos para a predição.  
    
    Parameters
    ----------
    month: int
    product: string
    revenue_b, revenue_c: float
    train_results_updated: dict
    
    Return
    ------
    int com a quantidade prevista em relação aos inputs dados.
    """
    
    dataset = train_results_updated[str(month)+'_'+product]
    point1 = np.array([revenue_b, revenue_c, pay_type])
    dataset_sz = dataset.shape[0]
    best_distance = np.inf
        
    # Busca pelo melhor centróide.
    for i in range(dataset_sz):
        point2 = dataset.iloc[i][['REVENUE_B', 'REVENUE_C', 'PAY_TYPE']].values
        distance = np.linalg.norm(point1 - point2)
        if distance < best_distance:
            best_distance = distance
            idx = i
            
    return dataset.iloc[idx]['QTY']


# Criamos um dicionário para guardar as predições por mês e produto.
predictions = {}
errors = []
cum_errors = []

for m in range(min_month, max_month+1):
    for p in products:
        data_example = data[(data['PROD']==p) & (data['MONTH']==m)]
        if (data_example.size!=0) and (str(m)+'_'+p in train_results_updated.keys()):
            i = 0
            for x in range(data_example.shape[0]):
                revenue_b, revenue_c, pay_type = data_example.iloc[i]['REVENUE_B'], data_example.iloc[i]['REVENUE_C'], data_example.iloc[i]['PAY_TYPE']
                predicted = int(prediction(m, p, revenue_b, revenue_c, pay_type, train_results_updated))
                actual = data_example.iloc[i]['QTY']
                errors.append(np.log(1+predicted) - np.log(actual))
                errors_tmp = np.array(errors)
                cum_errors.append(np.sqrt(np.mean(errors_tmp**2))) 
                predictions[str(m)+'_'+p] = [actual, predicted]
                i += 1

# Plot da evolução do RMSLE. Podemos ver que o modelo tende a melhorar conforme mais dados são 
# inseridos nele.
fig = plt.figure(figsize=[16, 6])
plt.plot(cum_errors, '*')
plt.xlabel('# observations')
plt.ylabel('RMSLE')
plt.show()

# Para finalizar, usamos o dicionário para criar um dataframe comparando as predições com os valores reais.
# Este dataframe é salvo no disco.
predictions = pd.DataFrame(predictions.values(), index=predictions.keys(), columns=['ACTUAL', 'PREDICTED'])
predictions.to_csv('predictions.csv')
print(predictions.head())

