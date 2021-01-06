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
#values go from 0 to 6, but no documentation available regarding that
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

m1 = orders.groupby('order_dow', sort=False)['order_id'].count().plot(kind="bar")
plt.ylabel('Quantity of orders')
plt.xlabel('')
plt.yticks(rotation=0)
plt.xticks(rotation=20)
plt.show()

n1 = m1.get_figure()
n1.savefig('week_days.png', bbox_inches='tight')

#Hour of the day
m2 = orders.groupby('order_hour_of_day')['order_id'].count().plot(kind="bar")
plt.ylabel('Quantity of orders')
plt.xlabel('Hour of the day')
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()

n2 = m2.get_figure()
n2.savefig('hour_of_the_day.png', bbox_inches='tight')

#Days since prior order
m3 = orders.groupby('days_since_prior_order')['order_id'].count().plot(kind="bar")
plt.ylabel('Quantity of orders')
plt.xlabel('Days since prior order')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

n3 = m3.get_figure()
n3.savefig('days_since_prior_order.png', bbox_inches='tight')

#Compare hour of the day by each day of the week to check for overlaps
orders = pd.read_csv('orders.csv') #reload csv
days = {0:'Saturday', 1:'Sunday', 2:'Monday', 3:'Tuesday', 
        4:'Wednesday', 5:'Thursday', 6:'Friday'}
orders['order_dow'] = orders['order_dow'].map(days) 
compare_days = {'Saturday':'Weekends', 'Sunday':'Weekends', 
                'Monday': 'Week days', 'Tuesday':'Week days', 
                'Wednesday':'Week days', 'Thursday':'Week days', 
                'Friday':'Week days'}
orders['order_dow'] = (orders['order_dow'].map(compare_days))

m4 = orders['order_hour_of_day'].hist(by=orders['order_dow']) #roughly the same distribution

#Create merge of orders and products
order1 = order_products_train.merge(products, how='inner')
order1 = order1.sort_values(by=['order_id','add_to_cart_order'])

order1['product_name'].value_counts().head(10).plot(
    kind='barh', figsize=(20,10), fontsize=12)
plt.title('Most sold products', fontsize = 20)
plt.ylabel('Product')
plt.xlabel('Number')
plt.show()

order2 = order1.groupby('order_id')['product_name'].agg(', '.join).reset_index()

