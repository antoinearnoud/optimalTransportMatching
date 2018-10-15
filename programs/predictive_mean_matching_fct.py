# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:40:57 2018

This file defines a PREDICTIVE MEAN MATCHING function.
It takes two DataFrames as input (dataset source and dataset target),
and two lists of variables.
The first is the variables shared by both datasets (used for the regression).
The second is the variables used to compute the total income (value to predict):
    it sumes up all these variables. If only one variable is provided in the list
    (because total income already in the source dataset), it is ok too.
The function returns a L by 3 matrix (L<=N+M-1) with the INDEX of each dataset
(column 1 and column 2) and the weight for this match (column 3).
One might want then import other variables from the original datasets, such that
an ids of the observations, other socio-economic characteristics. This can be
done using the INDEX of the original dataset.

@author: ARNOUD
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time # to compute time needed to run


def weighted_predictive_matching(df_t, df_s, shared_variables, 
                                 ytot_variables, weight_var = 'dweght', verbose = True):
    """returns a L by 3 panda df with indexes of observations and weights for matching
    >>> df_t = pd.DataFrame(np.array([[1,1,2],[1,2,1],[5,5,1]]), columns=['x1','x2', 'dweght'])
    >>> df_s = pd.DataFrame(np.array([[1,1,1,1],[1,2,3,1],[5,5,5,2]]), columns=['x1','x2','x3','dweght'])
    >>> shared_variables = ['x1','x2']
    >>> ytot_variables = ['x1','x2', 'x3']
    >>> df = weighted_predictive_matching(df_t, df_s, shared_variables, ytot_variables)
    >>> df['weight'].iat[0]
    1.0
    >>> df['id1'].iat[2]
    0
    """
    # copy datasets to work on because df_new = df just create a pointer, not a new dataset
    dataset1 = df_t.copy() # target dataset
    dataset2 = df_s.copy() # source dataset (donor)

    #checks that there is no missing weights
    if(dataset1[weight_var].isnull().sum() != 0):
        print("there are", dataset1[weight_var].isnull().sum(), "observations with missing values in weights in dataset 0")
    if(dataset2[weight_var].isnull().sum() != 0):
        print("there are", dataset2[weight_var].isnull().sum(), "observations with missing values in weights in dataset 1")

    #replace missing weights with zeros
    dataset1.weight_var = dataset1.weight_var.fillna(0)
    dataset2.weight_var = dataset2.weight_var.fillna(0)    

    # create data for regression from source dataset
    #dataset2['ytot'] = dataset2[ytot_variables].sum(axis=1) # NaN is treated as zero when summing, so ok.
    ytot = dataset2[ytot_variables].sum(axis=1) # NaN is treated as zero when summing, so ok.
    X = dataset2[shared_variables]
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    #y = dataset2['ytot']
    # do regression
    #model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
    model = sm.OLS(ytot, X).fit() 
    model.summary()

    # Compute predicted values
    ## Dataset source
    dataset2['ypredicted'] = model.predict(X)
    dataset2 = dataset2.sort_values(by = 'ypredicted', ascending = False) # will use the original index instead of a variable id.reset_index(drop = True) # no need of index because there is alredy id
    ## Dataset target
    X = dataset1[shared_variables]
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    dataset1['ypredicted'] = model.predict(X)
    dataset1 = dataset1.sort_values(by = 'ypredicted', ascending = False) # will use the original index of dataset instead of a varaible.reset_index(drop = True)

    # Compute transport matrix
    N = dataset1.shape[0]
    M = dataset2.shape[0]
    column_names = ['id1','id2', 'weight']
    # transform variales into integers ## SHOULDNT DO THAT. DON'T TRANSFORM IDENTIFIER
    #dataset1.dweght = dataset1.dweght.astype(int)
    #dataset2.dweght = dataset2.dweght.astype(int)
    #matchmatrix = pd.DataFrame(np.zeros((N + M, 3)), dtype = np.int64 , columns = column_names)

    # dont think you need to give the size of the matrix: start with empty matrix 
    # then append observations
    matchmatrix = pd.DataFrame(np.zeros((N + M, 3)), columns = column_names)
    #matchmatrix.id1 = matchmatrix.id1.astype(int)
    #matchmatrix.id2 = matchmatrix.id2.astype(int)

    #re-weighting
    weight1 = np.array(dataset1[weight_var])
    weight2 = np.array(dataset2[weight_var])
    # reweight to have same total weight (not what does CBO)
    weight2 = weight2 * dataset1[weight_var].sum() / dataset2[weight_var].sum()
    print("")
    print("total weights on dataset 0:  ", np.nansum(weight1))
    print("total weights on dataset 1:  ", np.nansum(weight2))
    # use numpy array for id variable because much faster than using dataset1[id_var][i] in the loop below! (divides time by 3)
    data1_index = np.array(dataset1.index)
    data2_index = np.array(dataset2.index)

   
    # matching
    start = time.time()
    i, j, k = 0, 0, 0
    while(i <= N - 1 and j<= M -1): # because i and j start at 0
    
         # how to deal with NaN??
         
        # match obs i with obs j
        matchmatrix['id1'].iat[k] = data1_index[i]; 
        matchmatrix['id2'].iat[k] = data2_index[j];
        # set weight of the match i<->j
        if (weight1[i] > weight2[j]):      
            matchmatrix['weight'][k]  = weight2[j];
            weight1[i] = weight1[i] - weight2[j];
            j = j + 1
        elif (weight1[i] < weight2[j]):
            matchmatrix['weight'][k] = weight1[i];
            weight2[j] = weight2[j] - weight1[i];
            i = i + 1
        else:
            matchmatrix['weight'][k] =  weight2[j];
            i = i + 1
            j = j + 1
        k = k + 1
        
    if(i < N-1):
        print("Warning...")
        print(N-1-i," obs on dataset 0 not used...")
    if(j < M-1):
        print("Warning...")
        print(M-1-j, " obs on dataset 1 not used...")
    matchmatrix = matchmatrix[matchmatrix.weight != 0]        
    end = time.time()
    if(verbose==True):
        print("")
        print("time needed for matching: ", end - start)    
        print("matching matrix dimension: ", matchmatrix.shape[0])
    
    # transform index into int
    matchmatrix.id1 = matchmatrix.id1.astype(int)
    matchmatrix.id2 = matchmatrix.id2.astype(int)
    
    return matchmatrix

if __name__ == "__main__":
    import doctest
    doctest.testmod()