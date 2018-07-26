# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:13:16 2018

@author: lud
"""
import tensorflow as tf
#tf.enable_eager_execution()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

def findUniqueFeatureValues(categorical_feature_col_names_, data: pd.DataFrame, dropna=False):
    categorical_feature_col_unique_values_ = {}
    for col_name in categorical_feature_col_names_:
        categorical_feature_col_unique_values_[col_name] = data[col_name].apply(str).fillna(
            'NA').unique().tolist()
        categorical_feature_col_unique_values_[col_name].sort()
    return categorical_feature_col_unique_values_

# load data
hist_exclude_long_breaks_df = pd.read_csv('../data/hist_exclude_long_breaks_df.csv', index_col = 0)


processing_rate_predictors = ['NbPacks'] + ['N' + s for s in list(map(str, list(range(1,21))))]
numeric_predictors = ["ORDER_LINE_COUNT", "TOTAL_QTY", "NbUnfinishedOrders", "simple_estimate"] + processing_rate_predictors
categorical_predictors = ["PUT_AREA", "VAS_REQUIREMENT", "H"]#, 'ORDER_STATUS'
target_variable = ["CYCLETIME_ADJUSTED"]
hist_exclude_long_breaks_df = hist_exclude_long_breaks_df[target_variable+categorical_predictors+numeric_predictors]
hist_exclude_long_breaks_df.H = hist_exclude_long_breaks_df.H.astype(str)
categorical_predictors_unique_values = findUniqueFeatureValues(categorical_predictors, hist_exclude_long_breaks_df)

dummy_categorical_predictors = []
for name, val_list in categorical_predictors_unique_values.items():
    dummy_categorical_predictors = dummy_categorical_predictors + [name + '_' + v for v in val_list]


#hist_exclude_long_breaks_df = pd.get_dummies(hist_exclude_long_breaks_df)    
train_data, test_data = train_test_split(hist_exclude_long_breaks_df, test_size=0.25, random_state = 123)

train_x, train_y = train_data, train_data.pop('CYCLETIME_ADJUSTED')
test_x, test_y = test_data, test_data.pop('CYCLETIME_ADJUSTED')

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(10000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

# Feature columns describe how to use the input.
my_feature_columns = []
for key in numeric_predictors:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
for key in categorical_predictors:
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=key,vocabulary_list = categorical_predictors_unique_values[key])    
    my_feature_columns.append(tf.feature_column.indicator_column(cat_col))
#Add a crossed column
cross_col = tf.feature_column.crossed_column(keys=categorical_predictors, hash_bucket_size=5)
my_feature_columns.append(tf.feature_column.indicator_column(cross_col))
# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.DNNRegressor(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[40,20,20,10],
    activation_fn=tf.nn.relu,
    dropout=0.01
    )

# Train the Model.
classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y,
                                             2000),
    steps=16000)

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, 1000))


predictions = classifier.predict(input_fn=lambda: eval_input_fn(test_x,labels=None,batch_size=100))
pred = [predictions.__next__()['predictions'][0] for i in range(test_y.shape[0])]
mean_absolute_error(test_y, pred)

#LEARNING_RATE = 0.001
#model_params = {"learning_rate": LEARNING_RATE}
#
#def model_fn(features, labels, mode, params):
#   # Logic to do the following:
#   # 1. Configure the model via TensorFlow operations
#   net = tf.feature_column.input_layer(features, params['feature_columns'])
#   for units in params['hidden_units']:
#      net = tf.layers.dense(net, units=units, activation=tf.nn.relu) 
#   output_layer = 
#   
#   
#   # 2. Define the loss function for training/evaluation
#   # 3. Define the training operation/optimizer
#   # 4. Generate predictions
#   return predictions, loss, train_op
#my_dnn = tf.contrib.learn.Estimator(model_fn = model_fn, params = model_params)
