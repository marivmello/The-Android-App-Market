import pandas as pd
import numpy as np
import plotly
import seaborn as seaborn

plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

apps_w_duplicates = pd.read_csv('datasets/apps.csv')
apps = apps_w_duplicates.drop_duplicates()
print('O número total de apps é {}'.format(apps))
print(apps.head())
chars_to_remove = ['+', ',', '$']
cols_to_clean = ['Installs', 'Price']
for col in cols_to_clean:
    for char in chars_to_remove:
        apps[col] = apps[col].apply(lambda x: x.replace(char, ''))
print(apps.info())
apps['Installs'] = apps['Installs'].astype(float)
apps['Price'] = apps['Price'].astype(float)
print(apps.info())

num_categories = len(apps['Category'].unique()) #contou a quantidade de categorias únicas
print('Número de categorias = ', num_categories)

num_apps_in_category = apps['Category'].value_counts() #contou o num de app em cada categoria
sorted_num_apps_in_category = num_apps_in_category.sort_values(ascending=False) #colocou os apps em ordem decrescente

data = [go.Bar(
    x = num_apps_in_category.index, y = num_apps_in_category.values)]
plotly.offline.iplot(data)

avg_app_rating = apps['Rating'].mean()
print('Average app rating = ', avg_app_rating)

data = [go.Histogram(x = apps['Rating'])]
layout = {'shapes': [{'type':'line', 'x0': avg_app_rating, 'y0': 0, 'x1': avg_app_rating, 'y1': 1000, 'line': {'dash':'dashdot'}}]}
plotly.offline.iplot({'data': data, 'layout': layout})

apps_with_size_and_rating_present = apps[(~apps['Rating'].isnull()) & (~apps['Size'].isnull())] #apenas as linhas com resposta
large_categories = apps_with_size_and_rating_present.groupby('Category').filter(lambda x: len(x) >= 250) #apenas aquelas categorias com no mínimo 250 apps
plt1 = sns.jointplot(x = large_categories['Size'], y = large_categories['Rating'])

paid_apps = apps_with_size_and_rating_present[apps_with_size_and_rating_present['Type']== 'Paid'] #apenas os apps pagos
plt2 = sns.jointplot(x = paid_apps['Price'], y = paid_apps['Rating'])

fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
#Selecionando algumas categorias
popular_app_cats = apps[apps.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE', 'LIFESTYLE', 'BUSINESS'])]
ax = sns.stripplot(x = popular_app_cats['Price'], y = popular_app_cats['Category'], jitter=True, linewidth=1)
ax.set_title('App princing trend across categories')
apps_above_200 = apps[apps['Price'] > 200] #Somente apps com preço acima de 200
print(apps_above_200[['Category', 'App', 'Price']])

apps_under_100 = popular_app_cats[popular_app_cats['Price']< 100] #Somente apps com preço abaixo de 100
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
ax = sns.stripplot(x = apps_under_100['Price'], y = apps_under_100['Category'], jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories after filtering for junk apps')

trace0 = go.Box(y=apps[apps['Type']=='Paid']['Installs'], name='Paid')
trace1 = go.Box(y=apps[apps['Type']=='Free']['Installs'], name='Free')
layout = go.Layout(title='"Number of downloads of paid apps vs. free apps',
                   yaxis=dict(title='Log number of downloads', type='log',
                              autorange=True))
data = [trace0, trace1]
plotly.offline.iplot({'data':data, 'layout':layout})

reviews_df = pd.read_csv('datasets/user_reviews.csv')
merged_df = apps.merge(reviews_df)
merged_df = merged_df.dropna(subset=['Sentiment', 'Review'])

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)

ax = sns.boxplot(x = merged_df['Type'], y = merged_df['Sentiment_Polarity'], data = merged_df)
ax.set_title('Sentiment Polarity Distribution')