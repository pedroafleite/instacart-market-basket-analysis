# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:43:14 2020

@author: Pedro
"""
#1) Import data

import pandas as pd
import matplotlib.pyplot as plt

aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
order_products_train = pd.read_csv('order_products__train.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
sample_submission = pd.read_csv('sample_submission.csv')

#2) Data exploration
orders.columns #seems to be good for exploring the time in which purchases were made
orders.dtypes
#Day of the week
orders.groupby('order_dow')['order_id'].count() #dow meaning 'day of the week'
#values go from 0 to 6, see documentation
#I would say that it starts on a saturday, since saturday and sunday should have the largest numbers (0 and 1 in this case)
#I argue that supported from experience of all else
days = {0:'Saturday', 1:'Sunday', 2:'Monday', 3:'Tuesday', 
        4:'Wednesday', 5:'Thursday', 6:'Friday'}
orders['order_dow'] = orders['order_dow'].map(days) 

orders.groupby('order_dow')['order_id'].count().plot(kind="bar")

#Reorder days of the week as shown in plot
dow = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
mapping = {order_dow: i for i, order_dow in enumerate(dow)}
key = orders['order_dow'].map(mapping)
orders = orders.iloc[key.argsort()]

orders.groupby('order_dow', sort=False)['order_id'].count().plot(kind="bar")

#Hour of the day
orders.groupby('order_hour_of_day')['order_id'].count().plot(kind="bar")

#Days since prior order
orders.groupby('days_since_prior_order')['order_id'].count().plot(kind="bar")

#Compare hour of the day by each day of the week to check for overlaps
compare_days = {'Saturday':'Weekends', 'Sunday':'Weekends', 
                'Monday': 'Week days', 'Tuesday':'Week days', 
                'Wednesday':'Week days', 'Thursday':'Week days', 
                'Friday':'Week days'}
orders['order_dow'] = (orders['order_dow'].map(compare_days))
orders['order_hour_of_day'].hist(by=orders['order_dow']) #roughly the same distribution

order1 = order_products_train.merge(products, how='inner')
order1 = order1.sort_values(by=['order_id','add_to_cart_order'])

order1['product_name'].value_counts().head(10).plot(
    kind='barh', figsize=(20,10), fontsize=12)
plt.title('Most sold products', fontsize = 20)
plt.ylabel('Product')
plt.xlabel('Number')
plt.show()

order1['add_to_cart_order'].count().plot(kind="bar")

order2 = order1.groupby('order_id')['product_name'].agg(', '.join).reset_index()

#4) Apriori algorithm
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

# get all shopping lists as one list
one_product = list(order2['product_name'].apply(lambda x: sorted(x.split(','))))

# instantiate transcation encoder
encoder = TransactionEncoder().fit(one_product)
onehot = encoder.transform(one_product)

# convert one-hot encode data to DataFrame
onehot = pd.DataFrame(onehot, columns=encoder.columns_)
# compute frequent items using the Apriori algorithm - Get up to three items
frequent_itemsets = apriori(onehot, min_support=.006, max_len=3, use_colnames=True)
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)

# compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# given that the left-hand side has two items, then which item is more likely to be added to the basket?
rules['lhs items'] = rules['antecedents'].apply(lambda x:len(x) )
rules[rules['lhs items']>1].sort_values('lift', ascending=False).head()
rules.to_csv('rules.csv', index=False)

#Data visualization
import seaborn as sns
# Replace frozen sets with strings
rules['antecedents_'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents_'] = rules['consequents'].apply(lambda a: ','.join(list(a)))
# Transform the DataFrame of rules into a matrix using the lift metric
pivot = rules[rules['lhs items']>=1].pivot(
    index='antecedents_', columns='consequents_', values= 'lift')
# Generate a heatmap with annotations on and the colorbar off
fig, ax = plt.subplots(figsize=(8,7))  
m = sns.heatmap(pivot, annot=True, linewidths=.1, annot_kws={"size":10}, ax=ax)
plt.ylabel('Antecedents')
plt.xlabel('Consequents')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

n = m.get_figure()
n.savefig('heatmap.png', bbox_inches='tight')
