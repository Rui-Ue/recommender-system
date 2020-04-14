
#%% README

# MovieLens の嗜好データを
# https://grouplens.org/datasets/movielens/100k/ 
# からダウンロードした．README に詳しい説明あり．ファイルとかカラムとか．
# https://yag.xyz/blog/2015/10/03/movielens-datasets/
# にもデータの簡単な紹介がある．いろいろなサイズのやつあって，とりあえず一番小さい(古い)やつにした．

# これを使って協調フィルタリングを(なるべくスクラッチで)実装してみる．





#%% import and check

import numpy as np
import pandas as pd

dat = pd.read_table("/Users/rui/data/ml-100k/u.data", header=None)
# dat
dat.columns = ["user_id", "movie_id", "rating", "timestamp"]

# dat.dtypes
# dat.isnull().sum()

# dat["user_id"].nunique()
# dat["movie_id"].nunique()
# dat["rating"].nunique()
# dat["rating"].value_counts()

# movie = pd.read_table("/Users/rui/data/ml-100k/u.item", header=None, encoding="shift_jis")
# movie = pd.read_table("/Users/rui/data/ml-100k/u.item_utf8.txt", header=None)
movie = pd.read_csv('/Users/rui/data/ml-100k/u.item_utf8.txt', sep='|', header=None)
movie_mst = movie[[0,1]]
movie_mst.columns = ["movie_id", "movie_name"]
# エンコーディングで苦労した．参考：https://stackoverflow.com/questions/30752973/encoding-issues-while-reading-importing-csv-file-in-python3-pandas
# カラム内容についてはREADMEを参照．



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

# ratemat  # 評価値行列．
# ratemat.isnull().mean()  # どの映画も9割近くが欠損

ranking = ratemat.notnull().mean().reset_index()
# mean(), sum(), notnull().mean()
# でいろんな観点からランキング見れる．
ranking.columns = ["movie_id", "value"]
ranking = ranking.merge(movie_mst, on="movie_id").sort_values("value", ascending=False)
ranking.iloc[:10,:]




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

# ratemat.index
# similarity(196, 186)
# similarity(196, 22)
# similarity(196, 925)
# similarity(196, 930)





#%% define class

class CollaborativeFiltering:
    """ class for Collaborative Filtering
    とりあえずユーザー間型メモリベース協調フィルタリングのみ実装．
    他のタイプはまた今度．
    """

    def __init__(self, sim_metric="pearson", nearest_k=10):
        """initialize
        Args:
          sim_metric: 類似度の指標．とりあえずピアソン相関係数のみ実装．
          nearest_k: k-NN の k．
        """
        self.sim_metric = sim_metric
        self.nearest_k = nearest_k

    def _calc_similarity(self, user1, user2, R, sim_metric):
        """類似度の計算
        メソッド fit() から呼び出して使う．
        """
        tf1 = R.loc[user1,:].notnull()
        tf2 = R.loc[user2,:].notnull()
        common_item = (tf1 & tf2)[(tf1 & tf2)].index
        sim = R.loc[user1, common_item].astype(int).corr(
            R.loc[user2, common_item].astype(int),
            method=sim_metric
        )
        return sim

    def fit(self, R):
        """ユーザー類似度行列を計算
        Args:
          R: 評価値行列．行がユーザー，列がアイテム．
        モデルベース協調フィルタとの関連を考えると，類似度計算がモデルフィットに対応．
        """
        self.sim_mat = pd.DataFrame(index=R.index, columns=R.index)
        for u1 in R.index:
            for u2 in R.index:
                sim = self._calc_similarity(
                    u1, u2, R, self.sim_metric
                )
                if np.isnan(sim):
                    sim = 0
                self.sim_mat.at[u1, u2] = sim
        self.R = R

    def predict(self, user, item):
        """評価値を予測
        Args:
          user: アクティブユーザー(レコメンド対象)
          item: 評価値を予測したいアイテム
        神嶌先生のpdfの(9.2)式を実装．
        user の k-NN に item を評価した人が一人もいない場合，
        user が付けた評価値の平均値で予測される．まあ直感的といえば直感的．
        """
        sim_all = self.sim_mat.loc[user, :].drop(index=user)
        sim_neighbor = sim_all[sim_all.abs().sort_values(ascending=False).iloc[:self.nearest_k].index]
        clb = 0
        for idx in sim_neighbor.index:
            avg = self.R.loc[idx, :].mean()
            rsd = self.R.loc[idx, item] - avg
            if not np.isnan(rsd):
                clb = clb + rsd * sim_neighbor[idx]
        clb = clb / sim_neighbor.abs().sum()
        return self.R.loc[user].mean() + clb

    def recommend(self, user, top_n, item_mst, key):
        """レコメンドするアイテムを決定
        Args:
          user: アクティブユーザー(レコメンド対象)
          top_n: 上位何個までのアイテムを返すか．
          item_mst: アイテムマスタ
          key: マスタとの結合キー(つまりアイテムIDのカラム名)
        """
        item_list = self.R.loc[user, self.R.loc[user, :].isnull()].index
        score = pd.Series(index=item_list)
        for item in item_list:
            score.at[item] = self.predict(user=user, item=item)
        out = score.sort_values(ascending=False).iloc[:10].reset_index()
        out.columns = [key, "predicted_score"]
        out = out.merge(item_mst, on=key, how="left")
        return out








#%% try collaborative filtering

colab_filt = CollaborativeFiltering(sim_metric="pearson", nearest_k=10)
# ハイパラ的な部分の設定．

ratemat_short = ratemat.iloc[:100, :]
# 計算時間きついので100人だけを対象にする．

colab_filt.fit(R=ratemat_short)
# モデルのフィット(=類似度の計算)

# colab_filt.sim_mat
# colab_filt.sim_mat.at[196, 22]

# ratemat_short
# ratemat_short.isnull().mean().sort_values()

colab_filt.predict(user=196, item=1)
# 評価値の予測．

colab_filt.recommend(user=196, top_n=10, item_mst=movie_mst, key="movie_id")
# 指定したユーザーに上位10個の映画をレコメンド
