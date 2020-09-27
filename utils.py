
import pandas as pd
import time

def make_feature(df_train, df_test):
    t2  = time.time()
    #敵味方

    #データ結合
    train_num = len(df_train)
    df = pd.concat([df_train.copy(), df_test.copy()])

    df = pd.merge(df, stage_area, how = "left", on = "stage")
    

    #カテゴリカル化
    #print(len(df))
    cat_cols = ["mode"]
    for c in cat_cols:
        vv, obj = pd.factorize(df[c])
        df[c] = vv
        df[c] = df[c].astype('category')

    #modeごとのweaponのcount
    df = count_x(df)

    #target encode
    df = target_x(df, train_num)


    #make features
    t3  = time.time()
    print("mkf:", round(t3-t2,1))
    return df[:train_num], df[train_num:]

def target_x(df, train_num, target) 
    #greedy target encode
    ts = df.groupby(["x"], as_index=False)["target"].mean()
    df = pd.merge(df, ts.rename(columns = {target:  "x"+ "_tgt"}), left_on ="x" , how = "left")
    return df

def count_x(df):
    df = df.assign(new_col_name=0).groupby(["x"])["new_col_name"].count().reset_index()
        #greedy target encode
    return df
