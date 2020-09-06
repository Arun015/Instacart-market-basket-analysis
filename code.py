import numpy as np 
import pandas as pd 
import tensorflow as tf
import os
import zipfile
zf = zipfile.ZipFile('order_products__prior.csv.zip')
af = zipfile.ZipFile('order_products__train.csv.zip')
bf = zipfile.ZipFile('orders.csv.zip')
cf = zipfile.ZipFile('products.csv.zip')

priors = pd.read_csv( zf.open('order_products__prior.csv'), dtype={'order_id': np.int32,'product_id': np.uint16,'add_to_cart_order': np.int16,
            'reordered': np.int8})
train = pd.read_csv( af.open('order_products__train.csv'), dtype={'order_id': np.int32,'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
orders = pd.read_csv(bf.open('orders.csv'), dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
products = pd.read_csv(cf.open('products.csv'), dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])
prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.int32)
prods['reorder_rate'] = (prods.reorders/prods.orders).astype(np.float32)
products = products.join(prods, on='product_id')
products.set_index('product_id', drop=False, inplace=True)
del prods
products.head()
users = users.join(usr)
del usr
orders.set_index('order_id', inplace=True, drop=False)
orders.head()

priors = priors.join(orders, on='order_id', rsuffix='_')
priors.head()
priors.drop('order_id_', inplace=True, axis=1)

usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)
usr.head()
# Creating features from the customer buying patterns
users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = users.all_products.map(len).astype(np.int16)
users.head()


users['average_basket'] = users.total_items/users.nb_orders
users.head()
### train / test orders ###
print('split orders : train, test')
test_orders = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']

train.set_index(['order_id', 'product_id'], inplace=True, drop=False)
train.head()
test_orders.head()
def features(selected_orders, labels_given=False):
    labels = []
    order_list = []
    product_list = []
    for row in selected_orders.itertuples():
        order_id = row.order_id
        #print(order_id)
        user_id = row.user_id
        user_products = users.all_products[user_id]
        #print(user_products)
        product_list += user_products
        # A list with order_id repeated len(user_products) time.
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]
        #print(labels)
       
    df = pd.DataFrame({'order_id':order_list, 'product_id':product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    
    #print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] =  df.user_id.map(users.average_basket)
    
    #print('order related features')
    # df['dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    #df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    
    #print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)
    return (df, labels)
df_train, labels = features(train_orders, labels_given=True)
import lightgbm as lgb
d_train = lgb.Dataset(df_train.astype(int), 
                     label=labels)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 50

print('light GBM train :-)')
bst = lgb.train(params, d_train, ROUNDS)
df_test, _ = features(test_orders)

print('light GBM predict')
preds = bst.predict(df_test)

df_test['pred'] = preds

TRESHOLD = 0.22  # guess, should be tuned with crossval on a subset of train data

d = dict()
for row in df_test.itertuples():
    if row.pred > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test_orders.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.head()
          
