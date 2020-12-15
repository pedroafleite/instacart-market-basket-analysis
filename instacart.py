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
arr4[:5]

#Implementation of the algorithm
from scipy.stats import multinomial, dirichlet

K = arr3[0]
X_test = arr3
alpha = 50/K
beta = 0.01
bic=0.01
gamma=50/K

class MultinomialExpectationMaximizer:
    def __init__(self, K, rtol=1e-3, max_iter=100, restarts=10):
        self._K = K
        self._rtol = rtol 
        self._max_iter = max_iter
        self._restarts = restarts

    def compute_log_likelihood(self, X_test, alpha, beta):
        mn_probs = np.zeros(X_test.shape[0])
        for k in range(beta.shape[0]):
            mn_probs_k = self._get_mixture_weight(alpha, k) * self._multinomial_prob(X_test, beta[k])
            mn_probs += mn_probs_k
        mn_probs[mn_probs == 0] = np.finfo(float).eps
        return np.log(mn_probs).sum()

    def compute_bic(self, X_test, alpha, beta, log_likelihood=None):
        if log_likelihood is None:
            log_likelihood = self.compute_log_likelihood(X_test, alpha, beta)
        N = X_test.shape[0]
        return np.log(N) * (alpha.size + beta.size) - 2 * log_likelihood

    def compute_icl_bic(self, bic, gamma):
        classification_entropy = -(np.log(gamma.max(axis=1))).sum()
        return bic - classification_entropy

    def _multinomial_prob(self, counts, beta, log=False):
        """
        Evaluates the multinomial probability for a given vector of counts
        counts: (N x C), matrix of counts
        beta: (C), vector of multinomial parameters for a specific cluster k
        Returns:
        p: (N), scalar values for the probabilities of observing each count vector given the beta parameters
        """
        n = counts.sum(axis=-1)
        m = multinomial(n, beta)
        if log:
            return m.logpmf(counts)
        return m.pmf(counts)

    def _e_step(self, X, alpha, beta):
        """
        Performs E-step on MNMM model
        Each input is numpy array:
        X: (N x C), matrix of counts
        alpha: (K) or (NxK) in the case of individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        Returns:
        gamma: (N x K), posterior probabilities for objects clusters assignments
        """
        # Compute gamma
        N = X.shape[0]
        K = beta.shape[0]
        weighted_multi_prob = np.zeros((N, K))
        for k in range(K):
            weighted_multi_prob[:, k] = self._get_mixture_weight(alpha, k) * self._multinomial_prob(X, beta[k])

        # To avoid division by 0
        weighted_multi_prob[weighted_multi_prob == 0] = np.finfo(float).eps

        denum = weighted_multi_prob.sum(axis=1)
        gamma = weighted_multi_prob / denum.reshape(-1, 1)

        return gamma

    def _get_mixture_weight(self, alpha, k):
        return alpha[k]

    def _m_step(self, X, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        X: (N x C), matrix of counts
        gamma: (N x K), posterior probabilities for objects clusters assignments
        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        alpha = self._m_step_alpha(gamma)

        # Compute beta
        beta = self._m_step_beta(X, gamma)

        return alpha, beta

    def _m_step_alpha(self, gamma):
        alpha = gamma.sum(axis=0) / gamma.sum()
        return alpha

    def _m_step_beta(self, X, gamma):
        weighted_counts = gamma.T.dot(X)
        beta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)
        return beta

    def _compute_vlb(self, X, alpha, beta, gamma):
        """
        Computes the variational lower bound
        X: (N x C), data points
        alpha: (K) or (NxK) with individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        gamma: (N x K), posterior probabilities for objects clusters assignments
        Returns value of variational lower bound
        """
        loss = 0
        for k in range(beta.shape[0]):
            weights = gamma[:, k]
            loss += np.sum(weights * (np.log(self._get_mixture_weight(alpha, k)) + self._multinomial_prob(X, beta[k], log=True)))
            loss -= np.sum(weights * np.log(weights))
        return loss

    def _init_params(self, X):
        C = X.shape[1]
        weights = np.random.randint(1, 20, self._K)
        alpha = weights / weights.sum()
        beta = dirichlet.rvs([2 * C] * C, self._K)
        return alpha, beta

    def _train_once(self, X):
        '''
        Runs one full cycle of the EM algorithm
        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        loss = float('inf')
        alpha, beta = self._init_params(X)
        gamma = None

        for it in range(self._max_iter):
            prev_loss = loss
            gamma = self._e_step(X, alpha, beta)
            alpha, beta = self._m_step(X, gamma)
            loss = self._compute_vlb(X, alpha, beta, gamma)
            print('Loss: %f' % loss)
            if it > 0 and np.abs((prev_loss - loss) / prev_loss) < self._rtol:
                    break
        return alpha, beta, gamma, loss

    def fit(self, X):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.
        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        best_loss = -float('inf')
        best_alpha = None
        best_beta = None
        best_gamma = None

        for it in range(self._restarts):
            print('iteration %i' % it)
            alpha, beta, gamma, loss = self._train_once(X)
            if loss > best_loss:
                print('better loss on iteration %i: %.10f' % (it, loss))
                best_loss = loss
                best_alpha = alpha
                best_beta = beta
                best_gamma = gamma

        return best_loss, best_alpha, best_beta, best_gamma


class IndividualMultinomialExpectationMaximizer(MultinomialExpectationMaximizer):
    def __init__(self, K, alpha_init, beta_init, household_ids, rtol=1e-3, max_iter=100, restarts=10):
        super().__init__(K, rtol, max_iter, restarts)
        self._household_ids = household_ids
        self._alpha_init = alpha_init
        self._beta_init = beta_init
        self._household_freqs = np.unique(household_ids, return_counts=True)[1]

    def _init_params(self, X):
        N = X.shape[0]
        alpha = np.vstack([self._alpha_init] * N)
        return alpha, self._beta_init

    def _get_mixture_weight(self, alpha, k):
        return alpha[:, k]

    def _m_step_alpha(self, gamma):
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
        gamma_df = pd.DataFrame(gamma, index=self._household_ids)
        grouped_gamma_sum = gamma_df.groupby(gamma_df.index).apply(sum)
        alpha = grouped_gamma_sum.values / grouped_gamma_sum.sum(axis=1).values.reshape(-1, 1)
        alpha = alpha.repeat(self._household_freqs, axis=0)
        return alpha
    
mem = np.vectorize(MultinomialExpectationMaximizer)
mem(arr2)
