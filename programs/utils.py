"""
Created on Mon Feb 12 17:11:14 2018

@author: Antoine Arnoud
"""

import math
import numpy as np
import pandas as pd
import ot
import sys
import time # used to compute time needed to run
from sklearn.linear_model import LinearRegression # to run the regression (predictive mean matching)


def replace_missing_weights_in_df(df_input, weight_var = 'dweght', inplace = False):
    """returns a dataframe with missing weights (NaN) replaced with zeros.
    This is inocuous in our work because an observation with a weight=0 is ignored.
    Parameters:
        df_input (DataFrame): DataFrame with weight variable
        weight_var (str): name of the weight variable; by default it uses dweght, as in Piketty, Saez, Zucman (2019)
        inpace (bool): if set to True, change are made inplace, so the DataFrame is modified
    Return:
        df (DataFrame): DataFrame with columns (weight_var) modified
    >>> df = pd.DataFrame(np.array([[1,1,2],[1,2,1],[5,5,np.NaN]]), columns=['x1','x2', 'dweght'])
    >>> df_output = replace_missing_weights_in_df(df)
    >>> df_output['dweght'][2]
    0.0
    >>> df_temp = replace_missing_weights_in_df(df, inplace = True) # use df_temp otherwise it prints the resut and shoud have >>> THE PRINTED df below.
    >>> df['dweght'][2]
    0.0
    """
    if inplace == True: # new pointer to df_input
        df = df_input
    else: # new version of the DataFrame
        df = df_input.copy()
    #check that there is no missing weights (isnull indicates if there are nan values)
    if(df[weight_var].isnull().sum() != 0):
        print("replacing ", df[weight_var].isnull().sum(), " missing values with zeros for weights in dataset")
    #replace missing weights with zeros
    df[weight_var] = df[weight_var].fillna(value=0) # either on a copy of df_input, or on df_input itself
    return df


# if want t ue array, but is not used. use same on df instead
def replace_missing_weights_in_array(weights_np):
    """returns a np.array with missing values replaced with zeros.
    Parameters:
        weights_np (np.array): array with np.NaN values
    Return:
        w (np.array): array with replaced missing values
    >>> w = np.array([5,5,np.NaN])
    >>> new_w = replace_missing_weights_in_array(w)
    >>> new_w[2]
    0.0
    >>> w[2]
    nan
    """
    w = weights_np.copy()
    #check that there is no missing weights (isnull indicates if there are nan values)
    if(np.isnan(w).sum() != 0):
        print("replacing ", np.isnan(w).sum(), " missing values with zeros for weights in dataset")
    #replace missing weights with zeros
    w = np.nan_to_num(w)
    return w


def normalize_weights(np_array1, np_array2, inplace = False):
    """normalize vectors of weights using total weights in first vector (np_array1).
    use np arrays (faster)
    Parameters:
        np_array1 (np.array): vector of weights 1 (used to normalize the other one)
        np_array2 (np.array): vector of weights 2 (weights will be normalized)
    Return:
        w1 (np.array): vector of weights 1 (identical to np_array1)
        w2 (np.array): vector of normalized weights (rescaling of np_array2)
    >>> np_array1 = np.array([1,5,4])
    >>> np_array2 = np.array([10,50,40])
    >>> w1,w2 = normalize_weights(np_array1, np_array2)
    >>> w1[0] # original values in np_array1 (integers)
    1
    >>> w2[0] # 1 * 10/100
    1.0
    """
    if inplace == True:
        print("select inplace=False, it is safer") # FIXME: to change later
        sys.exit()
    else:
        if np_array1.sum() != np_array2.sum():
            print('normalizing w2')
            w1 = np_array1.copy()
            w2 = np_array2.astype('float') # astype returns a copy
            w2 = w2 * w1.sum() / w2.sum()
        else:
            w1, w2 = np_array1.copy(), np_array2.copy()
    return w1, w2


def create_sum_vars(df_input, vars):
    ''' return pandas Series with the sum of variables from a DataSet.
    Parameters:
        df_input (DataFrame): DataFrame with variables to compues
        vars (list of strings): list of the variables' names to add
    Return:
        sum_series (pandas Series): pandas Series of sums.
    >>> df = pd.DataFrame([[10,20,30],[20,30,40]], columns = ['col1', 'col2', 'col3'])
    >>> sums = create_sum_vars(df, ['col1', 'col2'])
    >>> sums[0]
    30
    >>> sums[1]
    50
    '''
    for var in vars:
        if var not in df_input:
            sys.exit()
    sum_series = df_input[vars].sum(axis=1) # NaN is treated as zero when summing, so ok.
    return sum_series


#not very necessary, doesn't add anything, but create a function instead of a
def sort_df(df, col, inplace =  False):
    '''sort a DataFrame along one variable, with highest at the top (by default)
    Parameter:
        df (DataFrame): contains dataset
        col (str): variable used to sort the DataFrame
        inplace (bool): if True, change is inplace
    Return:
        df_out (DataFrame): DataFrame
    >>> df = pd.DataFrame([[5,5,5], [1,2,3],[20,30,40]], columns = ['col1', 'col2', 'col3'])
    >>> df_out = sort_df(df, 'col1')
    >>> df['col1'][0], df['col2'][0], df['col3'][0]
    (5, 5, 5)
    >>> df['col1'][1], df['col2'][1], df['col3'][1]
    (1, 2, 3)
    >>> df_out['col1'].iloc[0]
    20
    >>> df_out['col1'].iloc[2]
    1
    '''
    if inplace == False:
        # not sure this create a new dataframe. it seems so. FIXME: test
        df_out = df.sort_values(by = col, ascending = False, inplace = False) # preserves the index of the dataframe
    if inplace == True:
        df = df.sort_values(by = col, ascending = False, inplace = False)
    #df_out.reset_index(inplace = True, drop = True)
    return df_out

def ols(y, X):
    """fit regression (with contant?)
    Parameters:
        y (Pandas Series?): output
        X (DataFrame): regressors
    Return:
        lm (sklearn LinearRegression)
    """
    lm = LinearRegression()
    lm.fit(X,y)
    return lm


def run_predictions(lm, X):
    """Compute predicted values
    Parameters:
        lm (sklearnRegression object): model obtained using ols() fct
        X (DataFrame): regressors used for prediction
    Return:
        y_predicted (pandas Series ?): series of predicted values
    """
    y_predicted = lm.predict(X)
    return y_predicted


def vector_length_check(index1, index2, w1_input, w2_input):
    """Check that vectors are np.array and have right length. If not, exit.
    Parameters:
        index1 (np.array):
        index1 (np.array):
        w1_input (np.array) :
        w2_input (np.array) :
    Return:
        None
    """
    if (type(w1_input) is not np.ndarray or type(w2_input) is not np.ndarray):
        print("please give numpy arrays for weights")
        sys.exit()
    if  len(w1_input) != len(index1):
        print("index1 vector and weight1 vector are not same lengths.")
        sys.exit()
    if len(w2_input) != len(index2):
        print("index2 vector and weight2 vector are not same lengths.")
        sys.exit()
    return None

def balanced_problem_check(w1_input, w2_input, verbose = True):
    """Check that vectors have (approximately) the same total weight. If not, exit.
    Also transforms both array into float if one of them is (to be safe).
    Parameters:
        w1_input (np.array) :
        w2_input (np.array) :
    Return:
        w1 (np.array):
        w2 (np.array):
    """
    # floats: equality of total weights might not be exact (difficult with floats)
    if (not(issubclass(w1_input.dtype.type, np.integer)) or not(issubclass(w1_input.dtype.type, np.integer))):
        if verbose == True: print("some weights are non integers. working with float values.")
        w1, w2 = w1_input.copy().astype('float') , w2_input.copy().astype('float') # if one is integer and not the other, it is not possible to put float inside the vector of integer
        if not math.isclose(w1.sum(), w2.sum(), rel_tol = 1e-5, abs_tol=0.0):
            print("unbalanced problem")
            sys.exit()
    # integers only. looking for exact balanced samples.
    else:
        if verbose == True:
            print("all weights are integers.")
        w1, w2 = w1_input.copy(), w2_input.copy()
        if w1.sum() != w2.sum():
            print("unblanced problem. exit.")
            sys.exit()
    return w1, w2


#FIXME: can find a better name for this function. It is basically one-to-one matching, preserving weights.
def constrained_matching(index1, index2, w1_input, w2_input, verbose = True):
    """create a df with 3 columns: index1, index2, weight of matched obs.
    input must be 4 np.arrays for index number and weights
    Parameters:
        index1 (np.array): array of index (unique identifier)
        index2 (np.array): array of index (unique identifier)
        w1_input (np.array): array of weights
        w2_input (np.input): array of weights
        verbose (bool): if true, print stuff
    Return:
        result_df (DataFrame): 3 columns dataframes, id1, id2, weight
    >>> index1 = np.array([1,2,3])
    >>> index2 = np.array([3,2,1])
    >>> w1_input = np.array([2.0,1.0,1.0])
    >>> w2_input = np.array([1.0,1.0,2.0])
    >>> df = constrained_matching(index1, index2, w1_input, w2_input)
    >>> df['weight'].iat[1]
    1.0
    >>> df['id1'].iat[2]
    2
    >>> df['id2'].iat[2]
    1
    """
    if verbose == True:
        print('...start ranking algorithm...\n')
    vector_length_check(index1, index2, w1_input, w2_input)
    w1, w2 = balanced_problem_check(w1_input, w2_input, verbose = True)
    # define lengths
    N, M = len(w1), len(w2)
    # matching
    result_w, result_id1, result_id2 = [], [], []
    i, j, k = 0, 0, 0
    while(i <= N - 1 and j<= M -1): # use N-1 and M-1 (not N-1 and M-1) because i and j start at 0
        if verbose == True:
            print('\n......iteration: ', str(k))
            print('index df1: ' + str(i), "  w1 = " + str(w1[i]) )
            print('index df2: ' + str(j), "  w2 = " + str(w2[j]) )
        if (w1[i] > w2[j]):
            result_id1.append(index1[i])
            result_id2.append(index2[j])
            result_w.append(w2[j])
            w1[i] = w1[i] - w2[j]
            j += 1
        elif (w1[i] < w2[j]):
            result_id1.append(index1[i])
            result_id2.append(index2[j])
            result_w.append(w1[i])
            w2[j] = w2[j] - w1[i]
            i += 1
        elif(w1[i] == w2[j]):
            result_w.append(w1[i])
            result_id1.append(index1[i])
            result_id2.append(index2[j])
            i += 1
            j += 1
        else:
            print('problem. Cannot compare weights.')
            sys.exit()
        k += 1
    if(i < N-1):
        print("Warning... \n" +  str(N-1-i) + " obs on dataset 1 not used...")
    if(j < M-1):
        print("Warning... \n" + str(M-1-j) +  " obs on dataset 2 not used...")
    # create pandas DataFrame output, transform index into int
    result_df = pd.DataFrame(data = {'id1' : result_id1, 'id2' : result_id2, 'weight' : result_w})
    result_df['id1'], result_df['id2'] = result_df['id1'].astype(int), result_df['id2'].astype(int)
    return result_df


def drop_columns(df, columns = None, inplace = True):
    """ Drop list of columns from dataframe.
    Does not return error if list is empty or one column does not exist.
    >>> df = pd.DataFrame({'col_A': [1,2,3], 'col_B' : ['a','b','c']})
    >>> df_temp = drop_columns(df, ['col_A', 'OLEKSJDP'])
    """
    if inplace == False:
        sys.exit() # to change later
    if columns != None:
        for col in columns:
            if col not in df.columns:
                #sys.exit()
                print("column " + col + " not in dataset.")
                continue
            else:
                df.drop(col, axis=1, inplace=True)

# FIXME:
# PUT ALGOS BELOW IN THEIR OWN FILE (algos.py)

def ranking_algo(df1_input, df2_input, shared_vars_list, weight_var1 = 'dweght', weight_var2 = 'dweght', cols_drop1 = None, cols_drop2 = None, verbose = True):
    """Create a matched dataset using ranking algorithm
    Parameters:
        df1_input (np.array): dataset 1
        df2_input (np.array): dataset 2 (to merge with dataset 1)
        shared_vars_list (str or list of strings): column to match on (if list of str provided: sum up the values of all the columns)
        weight_var1 (np.input): array of weights
        weight_var2 (np.input): array of weights
        ...
        verbose (bool): if true, print stuff
    Return:
        result_df (DataFrame): synthetic data, with varaibles from both dataset (with suffix _1 and _2) and 3 columns (id1, id2, weight)
    >>> df1 = pd.DataFrame({'x1': [1,2,3], 'x2': [10,20,30], 'y':['small', 'medium', 'high'] , 'weight': [1,1,1]})
    >>> df2 = pd.DataFrame({'x1': [3,2,1], 'x2': [30,20,10], 'z': ['first', 'second', 'third'] ,'weight': [1,1,1]})
    >>> df_out = ranking_algo(df1, df2, ['x1', 'x2'], 'weight', 'weight')
    >>> df_out['z_2'].iloc[1]
    'second'
    """
    # create a new copy ( = is only ponter for DataFrames); can use inplace in functions below
    df1, df2 = df1_input.copy(), df2_input.copy()
    # check that index is unique (will use Pandas index as identifier for observations)
    if not (df1.index.is_unique and df2.index.is_unique): sys.exit()
    # replace missing values
    for dataframe in [df1, df2]:
        replace_missing_weights_in_df(dataframe, weight_var1, inplace = True)
    # create variable
    if ((len(shared_vars_list) == 1) and (shared_vars_list[0] in df1.columns) and (shared_vars_list[0] in df2.columns)): # shared_vars_list is a column existing in both datasets
        df1.sort_values(by = shared_vars_list, ascending = False, inplace = True)
        df2.sort_values(by = shared_vars_list, ascending = False, inplace = True)
    else: # sum up the variables and sort
        df1['y_tot'] = create_sum_vars(df1, shared_vars_list)
        df2['y_tot'] = create_sum_vars(df2, shared_vars_list)
        # sort along the varaible of insterst
        df1.sort_values(by = 'y_tot', ascending = False, inplace = True)
        df2.sort_values(by = 'y_tot', ascending = False, inplace = True)
    # normnalize weights
    w1, w2 = normalize_weights(np.array(df1[weight_var1]), np.array(df2[weight_var2]))
    # create identifier from index from DataFrames
    index1, index2 = np.array(df1.index), np.array(df2.index)
    # match observations
    output_df = constrained_matching(index1, index2, w1, w2, verbose = True)
    # Drop columns from dataset if needed
    drop_columns(df1, columns = cols_drop1, inplace = True)
    drop_columns(df2, columns = cols_drop2, inplace = True)
    #Join df1 and df2 to output_df
    #df2.columns = df2.columns.map(lambda x: str(x) + '_2')
    output_df = output_df.join(df1.add_suffix('_1'), on = 'id1') # rsuffix='_1';  'join' uses the index for df1 and the variable (column) id1 for output_df
    output_df = output_df.join(df2.add_suffix('_2'), on = 'id2')
    # can try:
    # output_df = output_df.join(df1.add_suffix('_1'), on = 'id1').join(df2.add_suffix('_2'), on = 'id2')
    # could drop id1 and id2 from output_df because comes from the index of the dataframe -> Keep because useful to check the function works
    # output_df.drop(['id1', 'id2'], axis=1, inplace=True)
    return output_df


def pmm_algo(df1_input, df2_input, shared_vars_list, to_sum_vars_list, weight_var1 = 'dweght', weight_var2 = 'dweght', cols_drop1 = None, cols_drop2 = None, verbose = True):
    """Create a matched dataset using ranking algorithm
    Parameters:
        df1_input (np.array): dataset 1
        df2_input (np.array): dataset 2 (to merge with dataset 1)
        shared_vars_list (str or list of strings): column to match on (if list of str provided: sum up the values of all the columns)
        weight_var1 (np.input): array of weights
        weight_var2 (np.input): array of weights
        ...
        verbose (bool): if true, print stuff
    Return:
        result_df (DataFrame): synthetic data, with varaibles from both dataset (with suffix _1 and _2) and 3 columns (id1, id2, weight)
    """
    # create a new copy ( = is only ponter for DataFrames); can use inplace in functions below
    df1, df2 = df1_input.copy(), df2_input.copy()
    # check that index is unique (will use Pandas index as identifier for observations)
    if not (df1.index.is_unique and df2.index.is_unique): sys.exit()
    # replace missing values
    for dataframe in [df1, df2]:
        replace_missing_weights_in_df(dataframe, weight_var1, inplace = True)
    # create variable
    df2['y_tot'] = create_sum_vars(df2, to_sum_vars_list)
    lm = ols(df2['y_tot'], df[[shared_vars_list]])
    df2['y_predicted'] = run_predictions(lm, df2[[shared_vars_list]])
    df1['y_predicted'] = run_predictions(lm, df1[[shared_vars_list]])
    # sort along the varaible of insterst
    df1.sort_values(by = 'y_predicted', ascending = False, inplace = True)
    df2.sort_values(by = 'y_predicted', ascending = False, inplace = True)
    # normnalize weights
    w1, w2 = normalize_weights((np.array(df1[weight_var1]), np.array(df2[weight_var2])))
    # create identifier from index from DataFrames
    index1, index2 = np.array(df1.index), np.array(df2.index)
    # match observations
    output_df = constrained_matching(index1, index2, w1, w2, verbose = True)
    # Drop columns from dataset if needed
    drop_columns(df1, columns = cols_drop1, inplace = True)
    drop_columns(df2, columns = cols_drop2, inplace = True)
    #Join df1 and df2 to output_df
    output_df = output_df.join(df1.add_suffix('_1'), on = 'id1').join(df2.add_suffix('_2'), on = 'id2')
    # could drop id1 and id2 from output_df because comes from the index of the dataframe (not very informative)-> Keep because useful to check the function works (drop later if want to)
    # output_df.drop(['id1', 'id2'], axis=1, inplace=True)
    return output_df


def opt_algo(df1_input, df2_input, shared_vars_list, to_sum_vars_list, weight_var1 = 'dweght', weight_var2 = 'dweght', cols_drop1 = None, cols_drop2 = None, verbose = True):
    """Create a matched dataset using ranking algorithm
    Parameters:
        df1_input (np.array): dataset 1
        df2_input (np.array): dataset 2 (to merge with dataset 1)
        shared_vars_list (str or list of strings): column to match on (if list of str provided: sum up the values of all the columns)
        weight_var1 (np.input): array of weights
        weight_var2 (np.input): array of weights
        ...
        verbose (bool): if true, print stuff
    Return:
        result_df (DataFrame): synthetic data, with varaibles from both dataset (with suffix _1 and _2) and 3 columns (id1, id2, weight)
    >>> df1 = pd.DataFrame({'x1': [1,2,3], 'x2': [10,20,30], 'y':['small', 'medium', 'high'] , 'weight': [1,1,1]})
    >>> df2 = pd.DataFrame({'x1': [3,2,1], 'x2': [30,20,10], 'z': ['first', 'second', 'third'] ,'weight': [1,1,1]})
    >>> df_out = ranking_algo(df1, df2, ['x1', 'x2'], 'weight', 'weight')
    >>> df_out['z_2'].iloc[1]
    'second'
    """
    # create a new copy ( = is only ponter for DataFrames); can use inplace in functions below
    df1, df2 = df1_input.copy(), df2_input.copy()
    # check that index is unique (will use Pandas index as identifier for observations)
    if not (df1.index.is_unique and df2.index.is_unique): sys.exit()
    # replace missing values
    for dataframe in [df1, df2]:
        replace_missing_weights_in_df(dataframe, weight_var1, inplace = True)
    # normnalize weights
    w1, w2 = normalize_weights((np.array(df1[weight_var1]), np.array(df2[weight_var2])))
    # create identifier from index from DataFrames
    index1, index2 = np.array(df1.index), np.array(df2.index)
    # match observations
    M = ot.dist(df1['shared_vars_list'], df2['shared_vars_list'], metric='euclidean')
    output_df = ot.emd(w1, w2, M)
    # Drop columns from dataset if needed
    #  ....
    # FIXME: check below...
    # .....
    drop_columns(df1, columns = cols_drop1, inplace = True)
    drop_columns(df2, columns = cols_drop2, inplace = True)
    #Join df1 and df2 to output_df
    output_df = output_df.join(df1.add_suffix('_1'), on = 'id1').join(df2.add_suffix('_2'), on = 'id2')
    # could drop id1 and id2 from output_df because comes from the index of the dataframe (not very informative)-> Keep because useful to check the function works (drop later if want to)
    # output_df.drop(['id1', 'id2'], axis=1, inplace=True)
    return output_df




# create submodules to create bins (I call them "classes").


if __name__ == "__main__":
    import doctest
    print = lambda *args, **kwargs: None  # does not output the print statements in doctest
    doctest.testmod()
