import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
#################### remember to force dataset to have class as name of class attribute
def build_data():
    unsplit_data_blood = pd.read_csv('blood.csv')
    train_set_blood,test_set_blood = train_test_split(unsplit_data_blood,test_size=0.33, random_state=42)
    train_set_blood.to_csv('train_set_blood.csv', index=False)
    test_set_blood.to_csv('test_set_blood.csv', index=False)

    unsplit_data_wine = pd.read_csv('wine.csv')
    train_set_wine,test_set_wine = train_test_split(unsplit_data_wine,test_size=0.33, random_state=42)
    train_set_wine.to_csv('train_set_wine.csv', index=False)
    test_set_wine.to_csv('test_set_wine.csv', index=False)
    
    unsplit_data_bank = pd.read_csv('bank.csv')
    train_set_bank,test_set_bank = train_test_split(unsplit_data_bank,test_size=0.33, random_state=42)
    train_set_bank.to_csv('train_set_bank.csv', index=False)
    test_set_bank.to_csv('test_set_bank.csv', index=False)


def run_exp():
   
    
    bank_train = pd.read_csv('train_set_bank.csv')
    bank_test = pd.read_csv('test_set_bank.csv')

    [bank_accuracy1_50,bank_features1_50] = make_and_test_forest(bank_train,bank_test,2,50)
    [bank_accuracy3_50,bank_features3_50] = make_and_test_forest(bank_train,bank_test,3,50)
    [bank_accuracy1_100,bank_features1_100] = make_and_test_forest(bank_train,bank_test,2,100)
    [bank_accuracy3_100,bank_features3_100] = make_and_test_forest(bank_train,bank_test,3,100)
    
    bank_res = [[bank_accuracy1_50,bank_features1_50],[bank_accuracy3_50,bank_features3_50],
    [bank_accuracy1_100,bank_features1_100],[bank_accuracy3_100,bank_features3_100]]
    
    
    for res in bank_res:
        f = open('bankres.txt','w')
        f.write('\n'+'acc'+repr(res[0]))
        f.write('feats'+'\n'+repr([1]))    
        f.close()


def make_and_test_forest(train_data,test_data,number_features,number_trees=50,fraction_samples=1):
    #from IPython.core.debugger import Tracer; Tracer()() 

    the_forest,features_used = plant_forest(train_data,number_features,number_trees,fraction_samples)
    accuracy,predictions,targets = make_prediction_forest(the_forest,test_data)

    return accuracy,features_used



def make_prediction_forest(forest,test_data):
    #list of lists where each nested list is the collection of predictions by the forest
    #import ipdb; ipdb.set_trace(context=20)
    print('make_prediction_forest'+'called')
    classes = test_data['class']
    classes = classes.reset_index(drop=True)
    
    forest_predictions = []
    for index,row in test_data.iterrows():
        tree_predictions = []
        for tree in forest:
            tree_predictions.append(make_prediction_tree(row,tree))
        
        tree_predictions_series = pd.Series(tree_predictions)
        predicted_class = tree_predictions_series.value_counts().index[0]    
        forest_predictions.append(predicted_class)
    
    forest_pred_series = pd.Series(forest_predictions)
    
    results = forest_pred_series==classes
    
    successes = 0
    
    for i in results:
        if i==True: successes+=1
    
    accuracy = successes/len(classes)     
   
    
    return accuracy,forest_pred_series,classes

    



def plant_forest(train_data,number_features=3,number_trees=50,fraction_samples=1,number_bins=20):
    '''build a random forest with supplied number of features, trees, bins and samples
    (fraction of training set must be <=1)'''
    print('plant_forest'+'called')
    
    #import ipdb; ipdb.set_trace(context=20)
    #resample training data for bagging
    resampled_training_sets = []
    
    for i in range(number_trees+1):
        dataset = train_data.sample(frac=fraction_samples,replace=True)
        resampled_training_sets.append(dataset) 

    the_forest = []
    features_used = []
    
    for dataset in resampled_training_sets:
        forest,features = plant_tree(dataset,number_features,number_bins)
        the_forest.append(forest)
        features_used = features_used+features
    
    feats = pd.Series(features_used).value_counts()
    return the_forest,feats




def make_prediction_tree(data_row,root_node):
    '''recursively traverse the tree from root to leaf turning left if feature value
    to test is less than dsplit_value or right otherwise until we reach a leaf node'''
    print('make_prediction_tree'+'called')
    
    #check if feature of data_row is less than dsplit_value else move to right branch
    if data_row[root_node['column_name']] < root_node['dsplit_value']:
        #check if at a branch or a leaf if branch recursively call predict else return leaf prediction
        if type(root_node['left']) is dict:
            return make_prediction_tree(data_row,root_node['left'])
        else:
            return root_node['left']
    else:
        if type(root_node['right']) is dict:
            return make_prediction_tree(data_row,root_node['right'])
        else:
            return root_node['right']                 

def plant_tree(train_data, number_features=3,num_bins=10):
    ''' start the process of recursion on the training data and let the tree
    grow to its max depth using subset of random features'''
      
    #get column names minus class
    #choose random set of features from column names
    data_column_names = list(train_data.columns)
    data_column_names.remove('class')
    random_features = random.sample(data_column_names,number_features)

    root_node = find_best_split_point(train_data,random_features)
    recursive_spltter(root_node,random_features)
    return root_node,random_features

def recursive_spltter(node,random_features):
    '''this function recursively splits the data starting with the root node which its passed
    untill the groups are homogenous or further splits result in empty nodes'''
    #from IPython.core.debugger import Tracer; Tracer()() 
    #retrieve two groups from the passed which is root or a recursive call on itself
    left_group,right_group = node['groups']
    #delete the groups entry in original node
    del node['groups']
    #check if the groups of the node are empty
    if left_group.empty or right_group.empty:
        #combine as we will use original to predict
        combined = pd.concat([left_group,right_group])
        predicted_class = combined['class'].value_counts().index[0]
        node['left']=node['right']=predicted_class
        return
    #check if the groups of the node are homogenous otherwise call recursive_spltter again
    if single_gini_index(left_group) == 0:
        predicted_class = left_group['class'].value_counts().index[0]
        node['left'] = predicted_class
    else :
        node['left'] = find_best_split_point(left_group,random_features)
        recursive_spltter(node['left'],random_features)

    if single_gini_index(right_group) == 0:
        predicted_class = right_group['class'].value_counts().index[0]
        node['right'] = predicted_class
    
    else:
        node['right'] = find_best_split_point(right_group,random_features)
        recursive_spltter(node['right'],random_features)

def find_best_split_point(passed_data, random_features):
    '''find best split point iterating over range of values returned from the 
    get_range_to_split_on function and return a dictionary which functions as a node '''

    
     #intialise values for best split point
    best_split_column = 'name'
    best_split_value = 0
    best_split_gini = 10
    best_split_groups = None
    
    #iterate over columns and rows searching for best split point
    for col_name in random_features:
        # for split_value in splitpoints_dict[col_name]:
        for index,row in passed_data.iterrows():
            left_split, right_split = build_split(passed_data,col_name,row[col_name])
            gini_score = multi_gini_index(left_split, right_split)

            if gini_score < best_split_gini:
                best_split_gini = gini_score
                best_split_column = col_name
                best_split_value = row[col_name]
                best_split_groups = left_split, right_split
    
    
    #print(best_split_column,best_split_gini)
    return {'column_name': best_split_column,'dsplit_value':best_split_value,
             'gini':best_split_gini, 'groups': best_split_groups}

def build_split(data,column_to_split,split_value):
    '''build 2 groups of data by splitting data on the column_to_split 
       at the split_value'''
    left_split = data[data[column_to_split]<split_value]
    right_split = data[data[column_to_split]>=split_value]
    
    return left_split,right_split

    
def multi_gini_index(group1,group2):
    '''Calculate Gini Impurity, func expects to be passed 
       the 2 groups of data that are the result of a split'''
    class_proportions_group1 = group1['class'].value_counts(normalize=True)    
    class_proportions_group2 = group2['class'].value_counts(normalize=True)    

    instance_proportion_group1 = len(group1)/(len(group1)+len(group2))
    instance_proportion_group2 = len(group2)/(len(group1)+len(group2))

    gini1 = (1 - class_proportions_group1.pow(2).sum())*(instance_proportion_group1)
    gini2 = (1 - class_proportions_group2.pow(2).sum())*(instance_proportion_group2)
    gini = gini1+gini2

    return gini

def single_gini_index(group):
    '''Calculate Gini Impurity of a single group'''
    class_proportions = group['class'].value_counts(normalize=True)    

    gini = (1 - class_proportions.pow(2).sum())
  
    return gini