def find_best_split_point(passed_data, num_bins=20):
    '''find best split point iterating over range of values for '''
    #get column names minus class
    data_column_names = data.columns[0:-1]
    splitpoints_dict = {}
    #get arrays of values for each to column to split on
    for column in data_column_names:
    	splitpoints_dict[column] = get_range_to_split_on(passed_data,column, num_bins)
    
    #intialise values for best split point
    best_split_column = 'name'
    best_split_value = 0
    best_split_gini = 10
    best_split_groups = None
    
    #iterate over columns and splitpoints searching for best split point
    for col_name in data_column_names:
    	for split_value in splitpoints_dict[name]:
    		left_split, right_split = build_split(passed_data,col_name,split_value)
    		gini_score = gini_index(left_split, right_split)

    		if gini_score < best_split_gini:
    			best_split_gini = gini_score
    			best_split_column = col_name
    			best_split_value = split_value
    			best_split_groups = left_split, right_split

    return {'column_name': best_split_column,'split_value':best_split_value,
             'gini':best_split_gini, 'groups': best_split_groups}

def build_split(data,column_to_split,split_value):
    '''build 2 groups of data by splitting data on the column_to_split 
       at the split_value'''
    left_split = data[data[column_to_split]<split_value]
    right_split = data[data[column_to_split]>=split_value]
    
    return left_split,right_split

    
def gini_index(group1,group2):
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
def get_range_to_split_on(data,column_to_split,num_of_points=20):
	'''returns desired number of equidistant points in a supplied pandas
	 dataframe column(series) numpy is expected and referenced by np, pandas by pd'''
    unique_values = data[column_to_split].unique()
    sorted_unique_values = np.sort(unique_values)
    values_for_splitting = np.linspace(sorted_unique_values[0],sorted_unique_values[-1],num_of_points)


