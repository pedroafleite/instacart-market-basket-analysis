# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:43:14 2020

@author: Pedro
"""

import pandas as pd
import matplotlib.pyplot as plt

aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
order_products_train = pd.read_csv('order_products__train.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
sample_submission = pd.read_csv('sample_submission.csv')

#Data exploration
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

order1 = order_products_train.set_index('product_id')
order2 = products.set_index('product_id')
order3 = order1.merge(order2, how='inner')

order3['product_name'].value_counts().head(10).plot(
    kind='barh', figsize=(20,10), fontsize=12)
plt.title('Products that are mostly put first into the masket', fontsize = 20)
plt.ylabel('Product')
plt.xlabel('Number')
plt.show()



