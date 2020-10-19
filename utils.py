
import pandas as pd
import time

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
    tgt_cols = ["count_uid",
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

