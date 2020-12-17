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

# 4) Apriori algorithm
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

# Data visualization of Market Basket Analysis
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

#by aisle
order3 = order1.merge(aisles, how='inner')
order3 = order3.groupby('order_id')['aisle'].agg(', '.join).reset_index()

# get all shopping lists as one list
one_product2 = list(order3['aisle'].apply(lambda x: sorted(x.split(','))))

# instantiate transcation encoder
encoder = TransactionEncoder().fit(one_product2)
onehot = encoder.transform(one_product2)

# convert one-hot encode data to DataFrame
onehot = pd.DataFrame(onehot, columns=encoder.columns_)
# compute frequent items using the Apriori algorithm - Get up to three items
frequent_itemsets = apriori(onehot, min_support=.006, max_len=3, use_colnames=True)
frequent_itemsets.to_csv('frequent_itemsets2.csv', index=False)

# compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# given that the left-hand side has two items, then which item is more likely to be added to the basket?
rules['lhs items'] = rules['antecedents'].apply(lambda x:len(x) )
rules[rules['lhs items']>1].sort_values('lift', ascending=False).head()
rules.to_csv('rules.csv', index=False)

# Data visualization of Market Basket Analysis
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
n.savefig('heatmap2.png', bbox_inches='tight')

# Examining Big Data processing issues
onehot.info(verbose=False, memory_usage="deep") #memory usage: 7.0 GB!

# 5) Multinomial mixture model
#Based on: https://towardsdatascience.com/multinomial-mixture-model-for-supermarket-shoppers-segmentation-a-complete-tutorial-268974d905da
# Data preparation
# Tree structure of the problem (actually, a forest): 
    #user_id -> order_id -> product_id
    
order3 = order1[['order_id', 'product_id']]
order3 = order3.merge(orders, how='inner')
order3 = order3[['user_id', 'order_id', 'product_id']]

#However, we do have departments and ailes
order4 = order1.merge(departments, how='inner')
order4.groupby('department')['order_id'].count().plot(kind="bar")
order4 = order4.merge(aisles, how='inner')
order4.groupby('aisle')['order_id'].count().plot(kind="bar")

# Sparse matrix
import numpy as np
from scipy import sparse

order5 = order4.apply(lambda s:s.astype("category"))
order5.aisle.cat.categories
arr = sparse.coo_matrix((np.ones(order5.shape[0]), 
    (order5.aisle.cat.codes, order5.order_id.cat.codes)))

#split numpy array in train and test datasets following the 80/20 rule
arr2 = arr.toarray()
arr2 = arr2.astype(int)
arr3 = pd.DataFrame.sparse.from_spmatrix(arr)

#pickling, so the work won't be lost

import pickle

with open('arr.pickle', 'wb') as f:
    pickle.dump(arr, f)    
with open('arr2.pickle', 'wb') as f:
    pickle.dump(arr2, f)
arr3.to_pickle('arr3.pickle')
order1.to_pickle('order1.pickle')
order2.to_pickle('order2.pickle')
order3.to_pickle('order3.pickle')
order4.to_pickle('order4.pickle')
order5.to_pickle('order5.pickle')

# How many products do clients usually buy?

def plot_customer_freq(x):
    basket_counts_by_customer = x.groupby(['order_id'])['product_id'].count()
    basket_counts_by_customer.plot.hist(bins=np.arange(100))

plot_customer_freq(order3)

# Make a sparse matrix with two indexes: user and order

order6 = order3.merge(products, how='inner')
order6 = order6[['user_id', 'order_id', 'product_name']]

baskets_data = order6.groupby(['user_id', 'order_id'])['product_name'].value_counts()
baskets_data_df = pd.DataFrame(data=baskets_data.values, index=baskets_data.index,
                               columns=['Count']).reset_index()
counts_df = baskets_data_df.pivot(index=['user_id', 'order_id'], columns=['product_name'],
                                  values=['Count'])['Count']
counts_df.fillna(0, inplace=True)



#Implementation of the algorithm

from tqdm import tqdm
from scipy.stats import multinomial, dirichlet

class MultinomialExpectationMaximizer:
    def __init__(self, K, rtol=1e-4, max_iter=100, restarts=10):
        self._K = K
        self._rtol = rtol
        self._max_iter = max_iter
        self._restarts = restarts

    def compute_log_likelihood(self, X_test, alpha, beta):
        mn_probs = np.zeros(X_test.shape[0])
        for k in range(beta.shape[0]):
            mn_probs_k = alpha[k] * self._multinomial_prob(X_test, beta[k])
            mn_probs += mn_probs_k
        mn_probs[mn_probs == 0] = np.finfo(float).eps
        return np.log(mn_probs).sum()
    
    def compute_aic(self, X_test, alpha, beta, log_likelihood=None):
        if log_likelihood is None:
            log_likelihood = self.compute_predictive_entropy(X_test, alpha, beta)
        return 2 * (alpha.size + beta.size) - 2 * log_likelihood

    def compute_bic(self, X_test, alpha, beta, log_likelihood=None):
        if log_likelihood is None:
            log_likelihood = self.compute_predictive_entropy(X_test, alpha, beta)
        N = X_test.shape[0]
        nb_params = (alpha.shape[0] - 1) + (beta.shape[0] * (beta.shape[1] - 1))
        return -log_likelihood + (0.5 * np.log(N) * nb_params)
    
    def compute_icl_bic(self, bic, gamma):
        classification_entropy = -(np.log(gamma.max(axis=1))).sum()
        return bic + classification_entropy

    def _multinomial_prob(self, counts, beta):
        """
        Evaluates the multinomial probability for a given vector of counts
        counts: (C), vector of counts for a specific observation
        beta: (C), vector of parameters for every component of the multinomial

        Returns:
        p: (1), scalar value for the probability of observing the count vector given the beta parameters
        """
        n = counts.sum(axis=-1)
        m = multinomial(n, beta)
        return m.pmf(counts)

    def _e_step(self, X, alpha, beta):
        """
        Performs E-step on MNMM model
        Each input is numpy array:
        X: (N x C), data points
        alpha: (K), mixture component weights
        beta: (K x C), multinomial categories weights

        Returns:
        gamma: (N x K), probabilities of clusters for objects
        """
        # Compute gamma
        N = X.shape[0]
        K = alpha.shape[0]
        weighted_multi_prob = np.zeros((N, K))
        for k in range(K):
            weighted_multi_prob[:, k] = alpha[k] * self._multinomial_prob(X, beta[k])

        denum = weighted_multi_prob.sum(axis=1)
        gamma = weighted_multi_prob / denum.reshape(-1, 1)

        return gamma

    def _m_step(self, X, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        X: (N x C), data points
        gamma: (N x K), probabilities of clusters for objects

        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        alpha = gamma.sum(axis=0) / gamma.sum()

        # Compute beta
        weighted_counts = gamma.T.dot(X)
        beta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)

        return alpha, beta

    def _compute_vlb(self, X, alpha, beta, gamma):
        """
        Each input is numpy array:
        X: (N x C), data points
        alpha: (K), mixture component weights
        beta: (K x C), multinomial categories weights
        gamma: (N x K), probabilities of clusters for objects

        Returns value of variational lower bound
        """
        loss = 0
        for k in range(alpha.shape[0]):
            weights = gamma[:, k]
            loss += np.sum(weights * (np.log(alpha[k]) + np.log(self._multinomial_prob(X, beta[k]))))
            loss -= np.sum(weights * np.log(weights))
        return loss
    
    def _init_params(self, C):
        alpha = np.array([1 / self._K] * self._K)
        beta = dirichlet.rvs([2 * C] * C, self._K)
        return alpha, beta

    def _train_once(self, X):
        loss = float('inf')
        C = X.shape[1]
        alpha, beta = self._init_params(C)

        for it in range(self._max_iter):
            prev_loss = loss
            gamma = self._e_step(X, alpha, beta)
            alpha, beta = self._m_step(X, gamma)
            loss = self._compute_vlb(X, alpha, beta, gamma)
            if it > 0 and (np.abs((prev_loss - loss) / prev_loss) < self._rtol):
                break
        return alpha, beta, gamma, loss

    def fit(self, X):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.

        X: (N, C), data points
        K: int, number of clusters
        '''
        best_loss = -float('inf')
        best_alpha = None
        best_beta = None
        best_gamma = None

        for it in range(self._restarts):
            alpha, beta, gamma, loss = self._train_once(X)
            if loss > best_loss:
                best_loss = loss
                best_alpha = alpha
                best_beta = beta
                best_gamma = gamma

        return best_loss, best_alpha, best_beta, best_gamma
    
def run_em(X, K_max=20, criterion='icl_bic'):
    
    if criterion not in {'icl_bic', 'bic'}:
        raise Exception('Unknown value for criterion: %s' % criterion)

    X = np.vstack(X)
    np.random.shuffle(X)

    nb_train = int(X.shape[0] * 0.8)
    X_train = X[:nb_train]
    X_test = X[nb_train:]

    likelihoods = []
    bics = []
    icl_bics = []
    best_k = -1
    best_alpha = None
    best_beta = None
    best_gamma = None
    prev_criterion = float('inf')
    
    for k in tqdm(range(2, K_max + 1)):
        model = MultinomialExpectationMaximizer(k, restarts=1)
        _, alpha, beta, gamma = model.fit(X_train)
        log_likelihood = model.compute_log_likelihood(X_test, alpha, beta)
        bic = model.compute_bic(X_test, alpha, beta, log_likelihood)
        icl_bic = model.compute_icl_bic(bic, gamma)
        likelihoods.append(log_likelihood)
        bics.append(bic)
        icl_bics.append(icl_bic)

        criterion_cur_value = icl_bic if criterion == 'icl_bic' else bic
        if criterion_cur_value < prev_criterion:
            prev_criterion = criterion_cur_value
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
            best_k = k
    
    print('best K = %i' % best_k)
    print('best_alpha: %s' % str(best_alpha))
    print('best_beta: %s' % str(best_beta))
    
    return likelihoods, bics, icl_bics, best_alpha, best_beta, best_gamma

import matplotlib.tri as tri
import matplotlib.lines as lines
from collections import defaultdict

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
AREA = 0.5 * 1 * 0.75**0.5
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    coords = np.array([tri_area(xy, p) for p in pairs]) / AREA
    return np.clip(coords, tol, 1.0 - tol)

def trimesh_coords_to_bucket_counts(trimesh, X):
    bucket_values_to_coord = {tuple(np.round(xy2bc(xy) * 16).astype(np.int)): xy for xy in zip(trimesh.x, trimesh.y)}
    coord_to_counts = defaultdict(int)
    for x in X:
        coord = bucket_values_to_coord[tuple(x)]
        coord_to_counts[coord] += 1
    counts = [coord_to_counts[xy] for xy in zip(trimesh.x, trimesh.y)]
    return counts

def plot_simplex():
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)

    fig, axes = plt.subplots(ncols=2, figsize=(16,7.1))
    axes = axes.ravel()
    for ax in axes:
        ax.axis('off')
    axes[0].triplot(trimesh, linewidth=1, color='darkgray')
    axes[1].triplot(trimesh, linewidth=1, color='darkgray')
    axes[1].set_title('Trinomial(n=16, β=[0.25, 0.5, 0.25])')
    
    tick_spacement = 1/16
    height = 0.75**0.5
    
    ax = axes[0]
    
    ax.text(-0.18, -0.12, 'X1', size=20)
    tick_x = 0
    tick_y = 0
    for i in range(17):
        if i % 2 == 0:
            ax.text(tick_x - 0.02, tick_y + 0.02, str(16-i), size=12)
        tick_x += height**2/16
        tick_y += 0.5*height/16

    dim_l1 = lines.Line2D([height**2, -0.07], [0.5*height, -0.04], linestyle='-', color='darkgray')
    ax.add_line(dim_l1)
    ax.scatter([-0.07], [-0.04], marker=(3, 0, 0), s=80, color='darkgray')
    
    # X2 axis
    ax.text(0.465, 1.04, 'X2', size=20)
    tick_y = 0
    for i in range(17):
        if i % 2 == 0:
            ax.text(0.51, tick_y + 0.01, str(i), size=12)
        tick_y += height/16
        
    dim_l2 = lines.Line2D([0.5, 0.5], [0, 0.98], linestyle='-', color='darkgray')
    ax.add_line(dim_l2)
    ax.scatter([0.498], [0.98], marker=(3, 0, 0), s=80, color='darkgray')

    # X3 axis
    ax.text(1.14, -0.12, 'X3', size=20)

    tick_x = 0.25
    tick_y = height/2
    for i in range(17):
        if i % 2 == 0:
            ax.text(tick_x + 0.01, tick_y + 0.01, str(i), size=12)
        tick_x += height**2/16
        tick_y -= 0.5*height/16

    dim_l3 = lines.Line2D([0.25, 1.1], [height/2, -(0.08/2**0.5)], linestyle='-', color='darkgray')
    ax.add_line(dim_l3)
    ax.scatter([1.1], [-(0.08/2**0.5)], marker=(3, 0, i*90), s=80, color='darkgray')


    return trimesh
    
def plot_trinomial(trimesh, x, color, n=None):
    n = n if n is not None else x.sum()
    counts = trimesh_coords_to_bucket_counts(trimesh, x)
    plt.scatter(x=trimesh.x, y=trimesh.y, color=color, 
                zorder=100,
                s=(np.array(counts) / n)*100000)

def plot_trinomials(X, colors):
    trimesh = plot_simplex()
    n = np.sum([x.sum() for x in X])
    for x, color in zip(X, colors):
        plot_trinomial(trimesh, x, color, n)

def make_dataset(n, alpha, beta):
    xs = []
    for k, alpha_k in enumerate(alpha):
        n_k = int(n * alpha_k)
        x = multinomial.rvs(n=16, p=beta[k], size=n_k)
        xs.append(x)
    return xs

alpha = [1/3]
beta = np.array([[0.25, 0.25, 0.5]])
X = make_dataset(10000, alpha, beta)
colors = ['coral']

plot_trinomials(X, colors)
plt.scatter([0.5], [((0.75**0.5)*0.5)], marker='*', s=100, color='red', zorder=100)

def plot_simplex():
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)

    fig, ax = plt.subplots(figsize=(8,7.1))
    ax.axis('off')
    plt.triplot(trimesh, linewidth=1, color='darkgray')
    
    ax.text(-0.09, -0.06, 'X1', size=20)
    ax.text(0.465, 0.9, 'X2', size=20)
    ax.text(1.07, -0.06, 'X3', size=20)
    
    tick_spacement = 1/16

    return trimesh

alpha = [1/3]
beta = np.array([[0.25, 0.25, 0.5]])
X = make_dataset(10000, alpha, beta)
colors = ['coral']

plot_trinomials(X, colors)
plt.scatter([0.5], [((0.75**0.5)*0.5)], marker='*', s=100, color='red', zorder=100)

def plot_em_run(likelihoods, ax):
    Ks = list(range(2, len(likelihoods) + 2))
    ax.scatter(Ks, likelihoods)
    ax.set_title('Likelihood by values of K')
    ax.set_ylabel('Likelihood')
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks)
    ax.set_xlabel('K')

def plot_simplex(ax):
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)

    ax.triplot(trimesh, linewidth=1, color='darkgray')
    ax.axis('off')
    
    ax.text(-0.09, -0.06, 'X1', size=20)
    ax.text(0.465, 0.9, 'X2', size=20)
    ax.text(1.07, -0.06, 'X3', size=20)
    
    tick_spacement = 1/16

    return trimesh

def plot_trinomial(trimesh, x, color, ax, z, n=None):
    n = n if n is not None else x.sum()
    counts = trimesh_coords_to_bucket_counts(trimesh, x)
    ax.scatter(x=trimesh.x, y=trimesh.y, color=color, 
               zorder=z,
               s=(np.array(counts) / n)*100000)

def plot_trinomials(X, colors, ax):
    trimesh = plot_simplex(ax)
    n = np.sum([x.sum() for x in X])
    z=100
    for x, color in zip(X, colors):
        plot_trinomial(trimesh, x, color, ax, z, n)
        z -= 1

def make_dataset(n, alpha, beta):
    xs = []
    for k, alpha_k in enumerate(alpha):
        n_k = int(n * alpha_k)
        x = multinomial.rvs(n=16, p=beta[k], size=n_k)
        xs.append(x)
    return xs

alpha = [0.1, 0.1, 0.8]
beta = np.array([[0.1, 0.1, 0.8], 
                 [0.1, 0.8, 0.1], 
                 [0.8, 0.1, 0.1]])
X = make_dataset(10000, alpha, beta)
colors = ('coral', 'forestgreen', 'purple')

likelihoods, bics, icl_bics, best_alpha, best_beta, best_gamma = run_em(X, criterion='bic')

fig, axes = plt.subplots(ncols=2, figsize=(16, 7.1))
axes = axes.ravel()
plot_trinomials(X, colors, axes[0])
plot_em_run(likelihoods, axes[1])
