import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import random
from sklearn.model_selection import ShuffleSplit

def load_df(file_name):
    'set name where datasets are found'
    ## accesses the .pkl file in the datasets folder,
    ## given the filename. file_name is of data type string.\
    ## a Data frame is returned.
    return pd.read_pickle('./datasets/'+file_name+'.pkl')

def sub_sample(df,sample_fraction):
    'randomly sample dataframe'
    ## Sample_fraction is a float =< 1.0
    ## df is a data frame containing the data we want randomly sample.
    sub=random.sample(range(len(df)),int(len(df)*sample_fraction))
    ##sub is a temporary list containing the randomly selected positions from df.
    ## The function returns a dataframe of randomly selected samples from the
    ## original data frame by randomly generating index positions for a given amount of sample fraction
    return df.iloc[sub]

## suggestions: consider using train size between 0-1.0 instead of int
## so that problem of data length is eliminated
def get_random_split(df):
    'splits data into train and test sets'
    ## df is a data frame that is to be split into train and test data sets.
    ## Shufflesplit is used to yield indices to split data into training and testing data
    rs=ShuffleSplit(n_splits=1,train_size=300,random_state=42)
    for i,j in rs.split(list(range(len(df)))):
        train_i,test_i=i,j
    ## The function returns a tuple with data frames, the first element is the train data
    ## the second element in the tuple is the test data
    return df.iloc[train_i],df.iloc[test_i]

#### Ask alex what is the use of cat_var
def explode_yield(df):
    '''seperate datapoints by IQ/SH yield
    df: seq  IQ_yield SH_yield
         1     10          20
    produces:
        seq  cat_var yield(y)
        1       [1,0]   10 
        1       [0,1]   20
    '''
    ## df is a data frame 
    OH_matrix=np.eye(2)
    ## creates a 2by2 matrix with a diagonal zero
    cat_var=[]
    exploded_df=[]
    ## cat_var , exploded_df are empty lists.   
    IQ_data=df[df['IQ_Average_bc'].notnull()]
    ## IQ_data is a data frame of all the seq that have non-null IQ yields

    if not IQ_data.empty:
        for i in range(len(IQ_data)):
            cat_var.append(OH_matrix[0].tolist())
        IQ_data.loc[:,'y']=IQ_data['IQ_Average_bc']
        IQ_data.loc[:,'y_std']=IQ_data['IQ_Average_bc_std']
        ## creating two new columns, y and y_std and putting the 
        ## IQ_Average_bc and IQ_Average_bc_std values from df in the respective colums
        exploded_df.append(IQ_data)

    SH_data=df[df['SH_Average_bc'].notnull()]
    ## SH_data is a data frame of all the seq that have non-null IQ yields
    
    if not SH_data.empty:
        for i in range(len(SH_data)):
            cat_var.append(OH_matrix[1].tolist())
        SH_data.loc[:,'y']=SH_data['SH_Average_bc']
        SH_data.loc[:,'y_std']=SH_data['SH_Average_bc_std']
        ## creating two new columns, y and y_std and putting the 
        ## SH_Average_bc and SH_Average_bc_std values from df in the respective colums
        exploded_df.append(SH_data)
    ## exploded_df is converted from a list of 2 data frames to a data frame containg its two elemets
    exploded_df=pd.concat(exploded_df,ignore_index=True)
    ## y is a list of the values from the y column in exploded_df. 
    y=exploded_df.loc[:,'y'].values.tolist()
    ## cat_var is a list of lists
    ##This function returns a tuple containing the data_frame exploded_df, the list cat_var and the list y 
    return exploded_df, cat_var, y

def explode_assays(assays,df):
    'seperate datapoints by assays to be predicted, similar to explode_yield'
    ## assays is either a tuple or a list containg non-repeating between 1-10. 
    ## each integer corresponds to a certain assay run. 
    ## As mentioned above df is the data frame containing pertinent data. 
    OH_matrix=np.eye(len(assays))
    OH_counter=0
    cat_var=[]
    exploded_df=[]
    ## OH_matrix is an identity matrix while cat_var and exploded_df are empty lists

    for i in assays:
        assay_df=df[df['Sort'+str(i)+'_mean_score'].notnull()]
        ## Accessing the assay information and creating a data frame for each not null value
        if not assay_df.empty:
            for j in range(len(assay_df)):
                cat_var.append(OH_matrix[OH_counter].tolist())
            assay_df.loc[:,'y']=assay_df['Sort'+str(i)+'_mean_score']
            ## creating new column, y and putting the assay mean score in that column.
            ## appending this updated data frame to exploded_df
            exploded_df.append(assay_df)
        OH_counter=OH_counter+1

    exploded_df=pd.concat(exploded_df,ignore_index=True)
    ## conccatinating the data frames in exploded_df into one data frame
    ## y is a list of the values in y column of the data frame.
    y=exploded_df.loc[:,'y'].values.tolist()
    
    ## the function returns a tuple with the exploded_df data frame, the cat_var list and the y list
    return exploded_df, cat_var, y

def mix_with_cat_var(x_a,cat_var):
    if len(cat_var[0])>1:
        x=[]
        for i in range(len(x_a)):
            x.append(x_a[i]+cat_var[i])
        return x
    else:
        return x_a #if there is only one catagory of 'y', dont include catagorical variable in model input

def get_ordinal(df):
    ## A getter function
    'sets ordinal encoded sequence to x_a in df for use in embedding models'
    ## df is a data frame
    x_a=df.loc[:,'Ordinal'].values.tolist()
    ## x_a is a list containg the values in the Oridinal column corresponding to each seq
    ## The values in the ordinal column is an array of float
    for i in range(len(x_a)):
        x_a[i]=x_a[i].tolist()
    ## converts arrays of floats into list of floats. 
    ## This function returns a list of lists containing floats, each float list corresponds to the ordinance column values
    return x_a


def get_onehot(df):
    ## A getter function
    'sets one_hot encoded sequence to x_a in df'
    ## df is a data frame
    x_a=df.loc[:,'One_Hot'].values.tolist()
    ## x_a is a list containg the values in the One_Hot column corresponding to each seq
    ## The values in the ordinal column is an array of float
    for i in range(len(x_a)):
        x_a[i]=x_a[i].tolist()
    ## converts arrays of floats into list of floats. 
    ## This function returns a list of lists containing floats, each float list corresponds to the One_Hot column values
    return x_a

def get_assays(assays,df):
    ## A getter function
    'sets assay values (of assays) to x_a in df'
    ## df is a data frame and assays is a list of non-repeating integers between 1-10
    ## column_names is an empty list
    column_names=[]
    for i in assays:
        column_names.append('Sort'+str(i)+'_mean_score')
    ## column_names is filled with string corresponding to heads of each assay column
    x_a=df.loc[:,column_names].values.tolist()
    ## x_a is a list of lists, with the interior list containg assay scores for each respective seq
    ## this function returns a list
    return x_a

def get_seq_and_assays(assays,df):
    'x_a is concat(onehot,assay scores)'
    assay_list=get_assays(assays,df)
    sequence_list=get_onehot(df)
    x_a=[]
    for i,j in zip(assay_list,sequence_list):
        x_a.append(i+j)
    return x_a

def get_control(df):
    ## A getter function
    'x_a should be null for a zero_rule model that guesses based upon average'
    x_a=[[]]*len(df)
    ## This function returns a list of empty lists equivalent as the same number of seqs
    return x_a

def get_embedding(df):
    ## A getter function
    'recreating what the above function is trying to do'
    ## x_a is a data frame containg only the learned_embedding colmn initally
    ## then it is turned into a list of lists, with each list containg floats. 
    x_a = df.loc[:,'learned_embedding']
    x_a = x_a.to_frame()
    x_a = x_a["learned_embedding"].tolist()
    for i in range(len(x_a)):
        x_a[i] = x_a[i][0][0]
    ## This function returns a list of float, each being the first entry
    ## in the list for each of the learned_embedding
    return x_a
