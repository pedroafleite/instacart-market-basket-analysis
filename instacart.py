# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:43:14 2020

@author: Pedro
"""

import pandas as pd

aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
order_products_train = pd.read_csv('order_products__train.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
sample_submission = pd.read_csv('sample_submission.csv')

#Data exploration
orders.columns #seems to be good for exploring the time in which purchases were made
