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


# 'sales' é o dataset da empresa B e 'comp' é o dataset da competição, constituída das empresas C1, ..., C6.
sales = pd.read_csv("sales.csv")
comp = pd.read_csv("comp_prices.csv")

# Criamos conexões e importamos os csv's para SQL, aí poderemos tratá-los como databases para usar o sqlite3.
# Apesar de ser possível trabalhar só com pandas, iremos fazer para ilustrar um trabalho com SQL.
conn_sales = sqlite3.connect('sales.db')
sales.to_sql('sales', con=conn_sales, if_exists='replace', index=False)
conn_comp = sqlite3.connect('comp.db')
comp.to_sql('comp', con=conn_comp, if_exists='replace', index=False)

# Criamos as classes 'cursor_sales' e 'cursor_comp' para executar as queries.
cursor_sales = conn_sales.cursor()
cursor_comp = conn_comp.cursor()

# Remoção dos outliers.
cursor_sales.execute("DELETE FROM sales WHERE REVENUE > 2980;")
cursor_comp.execute("DELETE FROM comp WHERE COMPETITOR_PRICE > 8100;")
conn_sales.commit()
conn_comp.commit()

# Agora podemos importar para csv, onde algumas colunas também foram retiradas. O motivo da retirada destas 
# colunas está no PDF.
sales_updated = pd.read_sql("SELECT * FROM sales;", conn_sales)
comp_updated = pd.read_sql_query("SELECT PROD_ID, DATE_EXTRACTION, COMPETITOR_PRICE FROM comp;", conn_comp)

# Fechamos a conexão com o database.
conn_sales.close()
conn_comp.close()

# Vamos padronizar o formato das datas, que será mês/dia, pois o ano é sempre 2015 e o horário não parece ser 
# muito relevante.
sales_updated['DATE_ORDER'] = sales_updated['DATE_ORDER'].str.replace('2015-', '')
comp_updated['DATE_EXTRACTION'] = comp_updated['DATE_EXTRACTION'].str.replace('2015-', '').str.split(' ').str[0]

# Padronizamos os nomes das colunas.
sales_updated = sales_updated.rename(columns={'DATE_ORDER': 'DATE', 'PROD_ID': 'PROD'})
comp_updated = comp_updated.rename(columns={'DATE_EXTRACTION': 'DATE', 'COMPETITOR_PRICE': 'REVENUE', 'PROD_ID': 'PROD'})


def fix_qty_order(df):
    """
    Arruma as quantidades em relação a um dado produto e dia. Fazemos um loop nos 
    rendimentos para data e produto fixados. A ideia é verificar se há repetições. 
    No caso positivo, somamos as repetições, guardamos a linha com a quantidade 
    atualizada e removemos as linhas antigas do dataframe.   
    Warning: esta função é lenta.
    
    Parameters
    ----------
    df: dataframe
    
    Return
    ------
    df: dataframe
        Dataframe após a organização das quantidades (coluna 'QTY_ORDER').
    """

    for date in df['DATE'].unique():
        for p in df['PROD'].unique():
            new_rows = []
            # Fazemos os agrupamentos necessários.
            df_group1 = df.groupby('PROD')
            df_group_p = df_group1.get_group(p)
            df_group2 = df_group_p.groupby('DATE')
            
            try:
                df_group_p_d = df_group2.get_group(date)
                
                for r in df_group_p_d['REVENUE'].unique():
                    sample = df_group_p_d[df_group_p_d['REVENUE']==r]
                    sample_qty = sample.shape[0]
                    if sample_qty > 1:
                        new_rows.append([p, date, sample_qty, r])
                        idx = df[(df['PROD']==p) & (df['DATE']==date) & (df['REVENUE']==r)].index
                        df = df.drop(idx)

                new_rows = pd.DataFrame(new_rows, columns=['PROD', 'DATE', 'QTY_ORDER', 'REVENUE'])
                df = pd.concat([df, new_rows], ignore_index=True, sort=False)
            
            except KeyError:
                pass
            
    return df


sales_updated = fix_qty_order(sales_updated)


def day_avrg_dataframe(out_df, seller):
    """
    Organizamos as datas, de modo que cada data seja repetida no máximo 9 vezes, onde cada 
    ocorrência representa a média da venda de algum produto naquele dia. Ou seja, dada uma
    data específica e um produto específico, pegamos todas as linhas com esta data e produto
    e as colapsamos numa única linha, com REVENUE sendo as médias das REVENUEs das linhas.
    
    Parameters
    ----------
    out_df: dataframe
    seller: string
        Deve ser 'B' ou 'C' (a competição).
        
    Return
    ------
    df: dataframe
        Dataframe com o mesmo formato de out_df, mas com as datas organizadas segundo o 
        procedimento descrito.
    """
    
    dates = out_df['DATE'].unique()
    date_groups = out_df.groupby('DATE')
    d = []
    
    if seller != 'B':
        for date in dates:
            b = date_groups.get_group(date) 
            c = {}
            for p in b['PROD'].unique():
                c[p] = b[b['PROD']==p]['REVENUE'].mean()
                d.append([p, date, c[p]])
    else:
        for date in dates:
            b = date_groups.get_group(date) 
            c = {}
            for p in b['PROD'].unique():
                c[p] = b[b['PROD']==p][['QTY_ORDER', 'REVENUE']].mean()
                c[p] = list(c[p].values)
                d.append([p, date, float(int(c[p][0])), c[p][1]])
    if seller != 'B':               
        df = pd.DataFrame(d, columns=['PROD', 'DATE', 'REVENUE_'+seller]).sort_values('DATE')
    else:               
        df = pd.DataFrame(d, columns=['PROD', 'DATE', 'QTY_ORDER', 'REVENUE_'+seller]).sort_values('DATE')
    return df


new_sellers_datasets = {}
new_sellers_datasets['B'] = day_avrg_dataframe(sales_updated, 'B')
new_sellers_datasets['C'] = day_avrg_dataframe(comp_updated, 'C')

# Criamos o novo data set.
data1 = new_sellers_datasets['B']
data2 = new_sellers_datasets['C']
data = data1.merge(data2, how='inner', on=['PROD', 'DATE'])

# O merge acima não remove algumas informações relevantes de 'sales'. Em particular, queremos que todo dia de 
# 'sales' tenha alguma informação correspondente em 'comp'. Na ausência de informação, iremos usar a própria 
# informação de 'sales' no lugar da informação da competição.
products = sales_updated['PROD'].unique()
sales_sz = sales_updated.shape[0]
comp_groups = comp_updated.groupby('PROD')
new_rows = []
comp_p_dates = {}
for p in products:
    comp_p = comp_groups.get_group(p)
    comp_p_dates[p] = comp_p['DATE'].unique()

for i in range(sales_sz):
    row = sales_updated.iloc[i]
    d = row['DATE']
    p = row['PROD']
    if d not in comp_p_dates[p]:
        new_rows.append(row.values)

df = pd.DataFrame(new_rows, columns=['PROD', 'DATE', 'QTY_ORDER', 'REVENUE_B'])
df['REVENUE_C'] = df['REVENUE_B']

data = pd.concat([data, df], ignore_index=True, sort=False)

# Por fim, será interessante ter uma coluna com os meses apenas.
data['MONTH'] = data['DATE'].str.split('-').str[0].astype('int')

# Normalização dos dados.
data_norm = data.copy()

s = data_norm['QTY_ORDER']
s = ( s - s.mean() )/( s.max() - s.min() )
data_norm['QTY_ORDER'] = s

for seller in ['B', 'C']:
    s = data_norm['REVENUE_'+seller]
    s = ( s - s.mean() )/( s.max() - s.min() )
    data_norm['REVENUE_'+seller] = s
    
# Para cada par (mês, produto) teremos um conjunto de clusters. Começamos agrupando os dados desta maneira.
train_groups = {}
min_month = min(data_norm['MONTH'].unique())
max_month = max(data_norm['MONTH'].unique())
for m in range(min_month, max_month+1):
    for p in products:
        train_groups[str(m)+'_'+p] = data_norm[(data_norm['MONTH']==m) & (data_norm['PROD']==p)]
        
# Usamos o método 'elbow' para determinar o número de clusters. 
train_results = {}


def get_results(kmeans, scores):
    """
    Esta função coleta os resultados do método KMeans, de modo que o número de cluster
    deve satisfazer abs(score[i]) < 1e-3, caso contrário assumimos 8 clusters.
    """
    
    L = len(scores)
    for i in range(L):
        if abs(scores[i]) < 1e-3:
            return kmeans[i].cluster_centers_
    return kmeans[-1].cluster_centers_


for m in range(min_month, max_month+1):
    for p in products:
        train_dataset = train_groups[str(m)+'_'+p]
        n_cluster = range(1, min(8, train_dataset.shape[0]))
        if train_dataset.size==0:
            pass
        else:
            # Sample é o dataframe relativo ao mês m e produto p.
            sample = train_dataset[train_dataset['PROD']==p][['QTY_ORDER', 'REVENUE_B', 'REVENUE_C']]
            # Aplicação do KMeans sobre sample.
            kmeans = [KMeans(n_clusters=i).fit(sample) for i in n_cluster]
            scores = [kmeans[i].score(sample) for i in range(len(kmeans))]
            # O número de clusters é o primeiro score satisfazendo abs(score[i]) < 1e-3.
            train_results[str(m)+'_'+p] = get_results(kmeans, scores)
    
    
# Para fazer previsões, vamos reverter a normalização.
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
    
    s = data['QTY_ORDER']
    df['QTY_ORDER'] = ( s.max() - s.min() )*df['QTY_ORDER'] + s.mean()

    for seller in ['B', 'C']:
        s = data['REVENUE_'+seller]
        df['REVENUE_'+seller] = ( s.max() - s.min() )*df['REVENUE_'+seller] + s.mean()
    return pd.DataFrame(df, columns=['QTY_ORDER', 'REVENUE_B', 'REVENUE_C'])


# train_results_updated é a versão não-normalizada de train_results. Se trata de um dicionário com keys no
# formato 'mês_produto', tais que train_results_updated['mês_produto'] é o dataframe dos centróides (obtidos 
# pelo KMeans) em relação ao par (mês, produto).
train_results_updated = {}
for m in range(min_month, max_month+1):
    for p in products:
        n_points = train_groups[str(m)+'_'+p].shape[0]
        if n_points != 0:
            df = pd.DataFrame(train_results[str(m)+'_'+p], columns=['QTY_ORDER', 'REVENUE_B', 'REVENUE_C']).copy()
            train_results_updated[str(m)+'_'+p] = denormalize(df, data) 
            train_results_updated[str(m)+'_'+p]['QTY_ORDER'] = train_results_updated[str(m)+'_'+p]['QTY_ORDER'].astype('int').copy()
            

def prediction(month, product, revenue_b, revenue_c, train_results_updated):
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
    point1 = np.array([revenue_b, revenue_c])
    dataset_sz = dataset.shape[0]
    best_distance = np.inf
        
    # Busca pelo melhor centróide.
    for i in range(dataset_sz):
        point2 = dataset.iloc[i][['REVENUE_B', 'REVENUE_C']].values
        distance = np.linalg.norm(point1 - point2)
        if distance < best_distance:
            best_distance = distance
            idx = i
            
    return dataset.iloc[idx]['QTY_ORDER']


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
                revenue_b, revenue_c = data_example.iloc[i]['REVENUE_B'], data_example.iloc[i]['REVENUE_C']
                predicted = prediction(m, p, revenue_b, revenue_c, train_results_updated)
                actual = data_example.iloc[i]['QTY_ORDER']
                errors.append(np.log(1+predicted) - np.log(actual))
                errors_tmp = np.array(errors)
                cum_errors.append(np.mean(errors_tmp**2)) 
                predictions[str(m)+'_'+p] = [actual, predicted]
                i += 1

# Plot da evolução do RMSLE. Podemos ver que o modelo tende a melhorar conforme mais dados são inseridos nele.
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
