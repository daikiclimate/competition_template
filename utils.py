import numpy as np
import time
import pandas as pd

def make_feature(df_train, df_test):
    t2  = time.time()
    #敵味方

    #データ結合
    train_num = len(df_train)
    df = pd.concat([df_train.copy(), df_test.copy()])


    #print(len(df))

    #modeごとのweaponのcount
    df = count_x(df)

    #target encoding
    tgt_cols = [
            "",
            ]
    df = target_encodes(df, train_num, tgt_cols)
 

    #make features
    t3  = time.time()
    print("mkf:", round(t3-t2,1))
    return df[:train_num], df[train_num:]

def target_encodes(df, train_num, tgt_cols):
    for c in tgt_cols:
        data_tmp = pd.DataFrame({c:df[:train_num][c], "target":df[:train_num].target})
        #validationの置換
        target_mean = data_tmp.groupby(c)["target"].mean()
        df.loc[train_num:, "tgt_" + c] = df[train_num:][c].map(target_mean)

        #学習データ
        tmp = np.repeat(np.nan, len(df[:train_num]))
        kf_encoding = KFold(n_splits = 4, shuffle = False, random_state = 0)

        # for train_index, valid_index in TimeSeriesSplit(n_splits=n_splits).split(np.arange(len(train))):
        for idx_1, idx_2 in kf_encoding.split(df[:train_num]):
            target_mean = data_tmp.iloc[idx_1].groupby(c)["target"].mean()
            tmp[idx_2] = df.loc[:train_num].loc[idx_2][c].map(target_mean)
        df[:train_num]["tgt_"+c] = tmp
    return df

def count_encodes(df, tgt_cols):
    for c in tgt_cols:
        df["cnt_" + c] = df[c].map(df.Age.value_counts())
    return df

def count_multi_encodes(df,tgt_cols):
    for c in tgt_cols:
        df["cnt_" + c[0] + "_" + c[1] = df.groupby([c[0], c[1]])["target"].transform("count")  # ←"Survived"は他の列名でもいい
    return df

def pivot_table_sample(df):
    data_give = pd.pivot_table(limit_df, index='cid2_target', columns='cid1_target', values='num_trades', aggfunc=np.sum).add_prefix("give_")
    div = np.tile(data_give.sum(axis = 1), (31,1)).T
    data_give = data_give/div

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                continue
                #NAlist.append(col)
                #df[col].fillna(mn-1,inplace=True)  

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df
 
