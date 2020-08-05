

# 10. 協調フィルタリング: モデルベース法





#### レコメンドシステムの分類 (復習)

- **内容ベースフィルタリング**：利用者プロファイルとアイテム特徴ベクトルを直接比較して、好みに合うものを推薦する。 (12章)
- **協調フィルタリング**：他のユーザーの嗜好を参考にして推薦する。
  - **メモリベース**：レコメンドの度に (モデル構築と) スコア算出を行う。(9章)
    - 利用者間型 (9.1節)
    - アイテム間型 (9.2節)
  - **モデルベース**：事前にモデルを構築しておいてレコメンドの時に使う。(10章)









# 10.1.  クラスタモデル





#### アイデア

1. アイテムへの評価値をもとに、ユーザーをクラスタリングする。
2. 同じクラスタ内のユーザーは嗜好パターンが似ている、と見なす。
3. 各ユーザーに、所属するクラスタの中で人気なアイテムを推薦する。





#### クラスタ数

- クラスタ数によって、推薦の質が大きく変わる。
- 小さく設定すると...
  - おおまかでパーソナライズされていない推薦がなされる。
  - cold-start 問題に対して比較的強い。（万人受けのアイテムを推薦できる）
- 大きく設定すると...
  - 高度にパーソナライズされた推薦がなされる。
  - cold-start 問題に対して比較的弱い。（クラスタ判定により多くのデータが必要）
  - 安定したクラスタを求めるのが難しくなる。





#### メリット・デメリット

メリットは、

- 実現が直感的で、実装も用意。
- モデルの構築が比較的高速。
- レコメンド時もユーザーと各クラスタ数の類似度を調べるだけなので、$O(K)$ で高速。

デメリットは、

- 複数の嗜好パターンが混在するユーザーに弱い。
  - 映画の場合、「アクションとサスペンスが好き」な人や「ホラーとサスペンスが好き」な人など、よく起こりうる。特定のジャンルだけを見る人はむしろ少ない。
  - 灰色の羊問題と呼ばれる。
  - （行列分解などのソフトクラスタリング系の方法で解決）





#### 共クラスタリング

- アイテムとユーザーを同時にクラスタリングする。
- [George and Merugu (2005) の A Scalable Collaborative Filtering Framework based on Co-clustering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.6458&rep=rep1&type=pdf) が参考になりそうなので読む。
- 共クラスタリングによって、レコメンド上どのようなアドバンテージが生まれる？
  - 精度？
  - 計算コスト (並列化しやすさ, システムへの乗せやすさなど) ？


