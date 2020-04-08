
#%% README

# MovieLens の嗜好データを
# https://grouplens.org/datasets/movielens/100k/ 
# からダウンロードした．

# これを使って協調フィルタリングを(なるべくスクラッチで)実装してみる．





#%% import and check

import numpy as np
import pandas as pd

dat = pd.read_table("/Users/rui/data/ml-100k/u.data", header=None)
dat
dat.columns = ["user_id", "movie_id", "rating", "timestamp"]

dat.dtypes
dat.isnull().sum()

dat["user_id"].nunique()
dat["movie_id"].nunique()
dat["rating"].nunique()
dat["rating"].value_counts()





#%% make Rating Matrix

ratemat = pd.DataFrame(
    index=dat["user_id"].unique(),
    columns=dat["movie_id"].unique()
)

for i in range(len(dat)):
    user = dat.iloc[i, 0]
    movie = dat.iloc[i, 1]
    rate = dat.iloc[i, 2]
    ratemat.at[user,movie] = rate

ratemat  # 評価値行列．
ratemat.isnull().mean()  # どの映画も9割近くが欠損




#%% calulate similarity

def similarity(user1, user2, method="pearson", check=True):
    tf1 = ratemat.loc[user1,:].notnull()
    tf2 = ratemat.loc[user2,:].notnull()
    if check:
      print((tf1 & tf2).sum())
    common_item = (tf1 & tf2)[(tf1 & tf2)].index
    sim = ratemat.loc[user1, common_item].astype(int).corr(
        ratemat.loc[user2, common_item].astype(int),
        method=method
    )
    return sim

ratemat.index
similarity(196, 186)
similarity(196, 22)
similarity(196, 925)
similarity(196, 930)





#%% predict rateing



