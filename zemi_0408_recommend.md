
# はじめに

<br>

最近興味を持っているレコメンドシステム(推薦システム, recommender systems)のご紹介．

行列分解などの機械学習・統計モデルは，レコメンドシステムのほんの一部分でしかない．なので，社会実装あるいは研究するにあたって，まず全体像を掴んでおく必要がある．

主に以下の2つを扱う．

- [神嶌先生のサーベイ](http://www.kamishima.net/archive/recsysdoc.pdf)：ボリュームもあり全体像を掴むのに良さそう．ただ，2008年以前(ディープラーニング等の登場前)の内容のみなので若干古い
- [神嶌先生のスライド](http://www.kamishima.net/archive/recsys.pdf)：図や例示が多く上記サーベイの補助資料として良さそう．ただ，講義資料をつなぎ合わせただけなので流れは配慮されていない．

このマークダウンの章立ては，上記サーベイの章立てに対応している．









<br>
<br>

# 目次について

<br>

「第Ⅰ部 推薦システムの概要」では，レコメンドシステムのモチベーションや目指すものが議論される．


「第Ⅱ部 推薦システムの実行過程」では，

1. 入力：ユーザーデータ・アイテムデータの取得
2. 処理：ユーザーの嗜好の予測
3. 出力：推薦の提示
    
といったシステム全体のフローが説明される．


「第Ⅲ部 推薦システムのアルゴリズム」では，
機械学習・統計モデルによる嗜好の予測について，少し詳しく説明される．
バックグラウンドによってはここから読み始めた方が理解しやすいかも．
（今日はここの途中まで）






<br>
<br>

# 第１章 推薦システム

<br>

レコメンドシステムの背景,モチベ,歴史について分かりやすく述べられている．簡単にまとめると，

- 大量の情報が発信されるようになった．
- → 誰もが大量の情報を得ることができるようになった．
- → 探している情報を特定できない．そもそも自分に合うような情報としてどのようなものがあるか分からない）．
- → 利用者にとって有用な情報を見つけ出すレコメンドシステムが考案された．



また，

> 推薦システムには大きく三つの要素技術が関連している．一つ目は，人間から必要な情報を収集し，人間との対話を扱うヒューマン・コンピュータ・インターフェース技術．二つ目は，収集したデータから推薦情報を生成し，それを目的に応じて変換する機械学習，統計的予測，そして情報検索の技術．三つ目は，推薦に必要な情報を蓄積し，処理し，流通させる基盤技術であるデータベース，並列計算，そしてネットワーク関連の技術．

とあるように，レコメンドは複合領域的な応用分野．
スライド26でも技術課題として「入力情報の高度化（自然言語処理，画像・音声認識，センサー情報）」が挙げられている．
また，次章で述べられるように，運用するにあたってはビジネス的感覚も重要．







<br>
<br>

# 第２章 推薦システムの分類と目的

<br>

2.1 〜 2.4 で，4種類の軸(基準)でレコメンドシステムを分類してくれる．




<br>

## 2.1 推薦の個人化の度合い

ここではレコメンドシステムを
「どのレベルまで個人化(パーソナライズ)してレコメンドするか」＝「どのくらい個々人に合わせてレコメンドを変える(オーダーメイドする)か」で分類している．


- 非個人化
  - Amazon の[人気ランキング](https://www.amazon.co.jp/ranking)
  - Apple Music の JPOP top 100
- 一時的個人化
  - Amazon の[よく一緒に購入されている商品](https://www.amazon.co.jp/%E7%94%BB%E5%83%8F%E8%AA%8D%E8%AD%98-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%83%95%E3%82%A7%E3%83%83%E3%82%B7%E3%83%A7%E3%83%8A%E3%83%AB%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E5%8E%9F%E7%94%B0-%E9%81%94%E4%B9%9F/dp/4061529129/ref=sr_1_2?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&dchild=1&keywords=%E7%94%BB%E5%83%8F%E8%AA%8D%E8%AD%98&qid=1586229381&sr=8-2)
  - YouTube の関連動画
- 永続的個人化
  - Amazon の[閲覧履歴に基づくおすすめ商品](https://www.amazon.co.jp/gp/css/homepage.html)
  - Apple Music の For You, Get Up! Mix


<br>

## 2.2 推薦システムの運用目的の分類

ここでは
「レコメンドシステムを運用する企業側の目的(目指すもの)」を基準に分類している．

実際は，分類というより「運用目的をいくつか紹介」みたいな感じになってる．


### 概要推薦

> こうした概要推薦は，システムを利用し始めたばかりか，ごくまれにしか利用しないような利用者を対象とする．これらの利用者が，自身の要求との関連性を見いだして，システムの利用を続けてもらえるように，大まかな情報を提供するのが目的である．

これがモチベ．

よって，ある程度のデータが必要な「永続的個人化」レベルのレコメンドは，この目的にあまり合わない．


### 利用者評価

> 他の利用者の方がずっと信用され，その推薦も受け入れられやすい．運用側には，システムに対する利用者の信頼を高めた
り，利用頻度を高めたりといった利点

これがモチベ．

分類とは言いつつも「概要評価かつ利用者評価」は成立しうると思う．「食べログの星の数でランキング」とか．曖昧に認識しておいた方が良さそう．


### 通知サービス

> これらの推薦は，システムの再利用を利用者に促すことを目的とする．

これがモチベ．

そう考えると，MLの例としてよく取り上げられるDM最適化も，レコメンドの一部とみなせる．



### 関連アイテム推薦

> 購入の判断を助けたり，購入の決断を促す場合や，補足的な商品などを提示して cross-selling（例：ハンバーガーにポテトなどの関連品を薦めて同時に購入させる）を促す

これがモチベ．



### 緊密な個人化

> 他のシステムとの差別化につながり，利用者の長期間にわたるロイヤリティ構築に役つ

これがモチベ．

これを目的とする場合「永続的な個人化」レベルのレコメンドをするしかない．

<br>

## 2.4 推薦システムの利用動機の分類


ここでは今度は
「レコメンドシステムを利用するユーザ側の目的(目指すもの)」を基準に分類している．

- 備忘録：食べログの「行ったお店」機能?
- 類似品：アマゾンの関連商品
- 新規アイテム：利用者がよく聞いてるアーティストの曲をレコメンド
- 視野を広げる：利用者がまだ知らないが好きそうなアーティストをレコメンド

スライド17に，

> 上の方ほど正解率を重視，下の方ほど多様性を重視

とあり面白い．詳しくは後述だが，正解率は「レコメンドしたものをユーザが好む確率」で多様性は「レコメンドされるジャンルの幅広さ」て感じ．


<br>

## 2.3 推薦システムの予測タスクの分類

2.2 と 2.4 は目的(目指すもの)による分類だった．ここでは目的を落とし込む「タスク」を基準に分類している．

ちょっと区別を理解しきれてない．


### 適合アイテム発見

「ユーザーが何かを探している(食べログでランチの店探してる)」というケースを想定し，それを当てにいく，というタスク．

なので ML 的には「分類」のタスク?


### 評価値予測

「ユーザーが(何か探しているわけではなく)アイテムのリストを眺めている」というケースを想定し，どのアイテムを閲覧するか等を決めるのを補助するために「ユーザーがアイテムにつけるであろう評価値」を予測する，というタスク．

なので ML 的には「回帰」のタスク?

身近な例はあまりないが，[MovieLens](https://movielens.org/) の MovieLens predicts for you がまさに「映画につけるであろう星の数の予測値」になってる．たしかにこれがあると，映画リストを眺めているときに便利だな．

「適合アイテム発見」と「評価値予測」の違いについては，スライド47も分かりやすい．


### 適合アイテム列挙

「ユーザがある条件のアイテムを網羅的に把握したい」というケースを想定し，それに合うアイテムを全て列挙する，というタスク．

具体的には，

> 例えば，会社の法務部門が関連する特許や判例を検索したり，スパムメールの可能性がないメールだけを閲覧したいといった場合である．

という記述がある．一般的に言われる「レコメンド」とはちょっと違って「情報検索」のようなイメージかな．

ML 的には「分類」のタスクだが，適合アイテム発見のときとは，重視すべき予測精度指標が異なってくる．後述．


### 効用最適化

適合アイテム発見タスクは「ユーザの好みに適合するような」アイテムを探すタスク．この「」の部分を一般化したものが効用最適化タスク．

例えば「ユーザがついでに買おうと思う」アイテムを探す，とかかな．
Amazonの「この商品買った人はこちらの商品を買ってますよ」みたいな．








<br>
<br>

# 第３章 推薦システム設計の要素

<br>

レコメンドシステムを設計する(入力データ・アルゴリズムを選ぶ)ためには，

- どのような運用上の制約のもとで
- どのような性能(性質)のレコメンドを目指すか 

を，目的に応じてある程度定めておく(把握しておく)必要がある．
要件定義的な．

そこで，

- レコメンドシステムの性能(性質)を測る評価指標は？→ 3.1節
- レコメンドシステムを運用する上でよくある制約は？→ 3.2節

<br>

## 3.1 推薦の性質

ML でもよくあるように，レコメンドシステムの目的・タスクに応じて，
重視すべき評価指標も異なってくる．

3.1.1の予測精度に関する指標以外にも，様々な側面からの指標がある．
レコメンド分野の特徴ともいえそう．

<br>

### 3.1.1 予測精度

レコメンドにおいて，

> 予測精度とは，予測して推薦したアイテムに，実際にどれくらい利用者が関心をもつかという規準

であり，予測精度に関する指標は色々ある．
ML で使うものと大体同じだが，目的に応じての使い分けが特徴的．

- 正解率(accuracy)：
分類問題とみなせる適合アイテム発見・列挙タスクにおいては，accuracy は定番の指標となる．

- 精度(precision)：
陽性的中率のこと．
適合アイテム発見タスクでは，ユーザの好みに合う(ユーザが探していて見つかりさえすれば買ってくれる)ものがレコメンドリストの中にちゃんと入ってるか(どのくらいの割合入ってるか)が気になる．
つまり precision が重要．
「好みにめっちゃ合いそうなやつ少しだけを推薦する(recall は気にせず precision を最大化)」っていう一見ずるい方法も全然あり．

- 再現率(recall)：
感度のこと．
適合アイテム列挙タスク(例えば関連特許を網羅的に列挙するタスク)では，ユーザの好みに合う(探している)ものをちゃんと全部網羅的にピックアップできるかどうかが気になる．
つまり recall が重要．
「ちょっとでも合いそうなやつは全部推薦する(precision は気にせず recall を最大化)」っていう一見ずるい方法も全然あり．

- 平均絶対誤差：回帰問題とみなせる評価値予測タスクにおいては，MAE は定番の指標．

- 順位相関：[スピアマンの順位相関係数](http://www.tamagaki.com/math/Statistics609.html)とか．評価値予測において，予測値と真値のズレ(MAEなど)よりも，予測した順位と真の順位のズレ(順位相関)が本質的に重要という考え方もある．なるほど，たしかに MovieLens の [top picks for you](https://movielens.org/explore/top-picks) みたいに予測値でソートするっていう見せ方あるしな．

また，レコメンドシステム(普通のMLでも言えることだが)で性能評価を行う方法として，オンラインとオフラインの2つがある．

- オンライン：実際にレコメンドシステムを運用してアイテムを推薦(モデルで予測)してみて，指標の値を算出し評価する．厳密だがコストが高い．

- オフライン：手元のデータでCVなどを行って指標の値を算出．コストは低いが厳密性に欠ける．例えば，精度評価時のアイテム分布と運用時のアイテム分布に違いがある可能性がある．具体的には，精度評価では何らかの評価(ユーザによる評価値付けや購買)が行われたアイテムしか使えないので不人気や新規のアイテムは対象外だが，運用では(データが溜まったりして)これらのアイテムがレコメンド対象になりうる．

<br>

### 3.1.2 多様性・セレンディピティ

レコメンドに特有の評価観点．ただ当てるだけじゃダメで...

> 多くの場合，利用者が知っているアイテムを推薦してもあまり有用ではない．よって，関心があることに加えて，推薦には，目新しさ (novelty)，すなわち，わかりきったものではないことが要求される．

ただ「あるユーザーがスピルバーグ監督のファンだとして，その人がまだ知らないスピルバーグ監督の新作をレコメンドした」という場合，
確かに novelty はあるけど，当たり前って感じ．レコメンドしないで放っておいても観に行くかもしれないし．
そこで...

> 推薦におけるセレンディピティ (serendipity) とは，この目新しさ (novelty) に，思いがけなさ，予見のできなさ，または意外性の要素が加わった概念である．

これを踏まえ，先ほどのスピルバーグ監督ファンに作風がよく似ている新人監督をレコメンドすれば，意外性(セレンディピティ)があって良い．

ただ，このような心理的な概念を定量的に評価するのは難しく，いろいろな研究が進んでいる．単純なものとしては，多様性＝「レコメンドリスト内の各アイテムがどれくらい類似していないか」で，セレンディピティを評価する．


> 一般に，利用者が推薦を採用したとき，その結果不満だったときのコストは低いが，満足したときの利得は大きい分野では，セレンディピティを重視すべきである．映画や音楽など娯楽に関する推薦では，こうした状況になることが多い．

このように，ビジネス的な観点から考えてどういう性能性質(評価指標)を重視すべきか判断するべき．

<br>

### 3.1.3 被覆率

これもレコメンド特有かもしれない．

被覆率(coverage)とは，全アイテムのうち評価値(ユーザへの適合度)の予測が可能なアイテムの割合，すなわちレコメンドの候補に入るアイテムの割合．

> 適合アイテム発見タスクでは，利用者が満足するものが何か見つかれば良いので被覆率は比較的低くても問題は生じない．評価閲覧が目的なら，評価値のないアイテムが多数あるのは不便なので被覆率は高くあるべきである．適合アイテム列挙タスクでは，推薦すべき対象の見落としは許されないので，基本的に被覆率は 100% でなければならない．

このように，coverage の重要度もタスク(目的)に応じて変わってくる．

用いるモデル・アルゴリズムによって cover できるアイテムの条件は変わってくるので，coverage の重要度に応じた使い分けが重要．

<br>

## 3.2 推薦候補の予測に関する制約

運用に耐えうるレコメンドシステムを設計する(入力データ・アルゴリズムを選ぶ)ためには「どのような性能性質のレコメンドを目指すか(6.1節)」に加えて「どのような制約があるか，どのような制約を我慢し守る必要があるか」を考えるべき．

<br>

### 3.2.1 嗜好データの制約

> 嗜好データの最も顕著な特徴は非常に疎 (sparse) であることである．
すなわち，非常に多くのアイテムが存在するが，利用者が評価しているのはごく一部で，その他のアイテムへの評価値は欠損している．

ここでいう疎(sparse)ってのは，スパース推定の疎(ほとんど0/重要な情報は一部しかない)とは違って，めっちゃ欠損があってスカスカっていうニュアンス．

> 最後に嗜好データの更新の問題がある．推薦システムは運用中に，随時嗜好データが追加される．また，新たに利用者やアイテムがデータベースに追加されることもある．こうした変化に応じて予測モデルを更新する必要がある

これはレコメンドの結構本質的な課題な気がする．
この更新をどのくらいの頻度で行いたいかが，後述の様々なモデル・アルゴリズムのうちどれを使うべきかに関わってくる．

<br>

# 3.2.2 その他の制約や条件

> 10～1000 の要求に対して，10～100 ミリ秒の時間で応答することが要求される [Ben Schafer 01, Linden 03]．このような高いスケーラビリティを達成しつつ，正確に予測することも困難な課題

これも実際にサービスとして運用する上では必須の制約．
ちなみにスケーラビリティとは「利用者や仕事の増大に適応できる能力・度合い」の意．








<br>
<br>

# 第４章 推薦システムの実行過程

<br>

ここまでは，レコメンドシステムの

- モチベーション(なぜレコメンドしたいか)
- 分類(どういう種類があるか)
- 性能評価指標(どういうシステムを目指すか)
- 制約(どういう制限を満たす必要があるか)

を扱ってきた．ここからは具体的な設計・実装方法．

「O-I-Pモデル」は普通のシステムと同じ
「入力を受け取って何らかの処理をして出力を返す」というモデル．

推薦システムを実装する時には，O-I-Pモデルの

- Input：データの入力(どういうデータを取得して使うか)
- Process：嗜好の予測(どういうアルゴリズムで推薦するアイテムを決めるか)
- Output：推薦の提示(どういう見せ方でレコメンドするか)

の3段階で考えると整理しやすい．
それぞれ5章,6章,7章で詳説される．
「嗜好の予測」で機械学習・統計モデリングが活躍し，他2つでは心理統計・行動経済学などが関連しそう．


> 推薦を受けようとしている人を活動利用者 (active user) と呼ぶ．

一般的な「アクティブユーザー」という言葉と意味が若干違う．今まさにアイテムをレコメンドしてあげようとしている人，という感じ．定式化する時に便利．






<br>
<br>

# 第5章 データの入力

<br>

嗜好データ(利用者の各アイテムへの関心の度合いを数値化したもの)やユーザー属性情報の他にも，検索質問(食べログの料理ジャンルや想定シチュエーションみたいな)も入力データと見なす．が，

> 検索質問は，情報検索やデータベースのクエリ検索の技術がほぼ転用できる

ので，この章では嗜好データについて扱う．




<br>

## 5.1 暗黙的と明示的な嗜好データの獲得

> 嗜好データを獲得するアプローチは，おおきく暗黙的と明示的の二種類に分けられる．明示的な獲得とは，利用者に好き嫌いや，関心のあるなしを質問し，利用者に回答してもらう方法である．もう一方の暗黙的な獲得とは，利用者の行動をから，利用者の嗜好や関心を推察することで嗜好データを得る方法である．

この定義はわかりやすい．


明示的な獲得の例としては，
- [MovieLens](https://movielens.org/)でユーザーに映画に対して星を付けてもらう．
- Apple Music 始めた時に，好きなジャンルを聞かれる．

暗黙的な獲得の例としては，
- Amazon でユーザーがクリックしたアイテムを記録しておく
- アイテムを眺めていた時間も記録しておく


> 二つの嗜好データの獲得法を比較する．これらの獲得法の長所と短所を表 5.1にまとめた．

この表と本文の説明が十分分かりやすい．


<br>

## 5.2 明示的な獲得

> 明示的な獲得法では，利用者はアイテムを評価することを面倒だと思うので，暗黙的な方法に比べて多数の嗜好データを集めにくい

「明示的な獲得ではユーザーに能動的に評価してもらう必要があるが，どうやって動機づければ良い？」というリサーチクエスチョンに対して，いろいろな方法が研究されている．行動経済学っぽい話．


また，心理統計やアンケート調査・設計と同じような問題意識もある．「どのような質問をして回答をどう処理すればユーザの真の嗜好を測れるだろうか？」という話が，5.2.1 〜 5.2.3 で扱われていく．



<br>

### 5.2.1 採点法と格付け法

- 採点法：数字で答えさせる．食べログの星の数など．
- 格付け法：ランクで答えさせる．YouTubeの高評価ボタンと低評価ボタンなど．

この分類とは別の問題として，得られた結果を間隔尺度と見なすのか順序尺度と見なすのか，も重要そう．採点法で取得したら感覚尺度，格付け法で取得したら順序尺度，という風な単純な解釈は無理．「星1つと2つの間隔」と「星2つと3つの間隔」が同じ価値とは限らないので．

↑について，5.2.3に，次のような記述が出てきてた．
> 採点法や格付け法で得られる量は，本質的には大小関係にのみ意味がある順序尺度 [Stevens 51, 鷲尾 98] であると指摘されている [中森 00]

どういう質問と選択肢を用意すれば良いか，というアンケート設計と同じような研究が色々されている．


<br>

### 5.2.2 評価値の揺らぎや偏り

（「揺らぎ」「偏り」とかの言葉の使われ方がイマイチよく分からなくて，うまく理解・整理できない．）

とりあえず採点法と格付け法の問題点として，

- 真の嗜好と評価値が様々な要因(質問内容,質問時期,性格)によって乖離する．
- 人気商品にばかり評価値が付けられて「人気商品だけで精度評価(学習も?)したモデルで全商品をレコメンドする」みたいな状況が発生する．

などがあるのかな．真の嗜好をどう定義するかってのもハッキリしていないので，うまく理解できない．


<br>

### 5.2.3 順序の利用

> 利用者ごとの平均評価値を 0 に正規化することで予測精度が向上することを報告している．このことは，採点法で得た評価値の絶対的な値ではなく，相対的な大小が重要であることを示唆しているといえるだろう．

これはまあ納得できる．Aさんの星1とBさんの星1の価値(嗜好の度合い)の違いが除去されたからだろう．

> 好きなものから嫌いなものへ順に，複数の対象を並べるという順位法 (ranking method) を利用する「なんとなく協調フィルタリング」[Kamishima 03b, Kamishima 06] を神嶌は提案した．少なくとも調査したデータにおいて，順位法の採用で予測精度が向上した．

これも納得．採点法・格付け法の代わりに順位法(好きなものから順に並べてください)で嗜好を測った方が確かに筋が良さそう．

（ただ，具体的になぜこれで精度が上がっているのか，採点法・格付け法のどういう問題点を改善したのか，真の嗜好とのズレが小さくなったのか，真の嗜好はどう定義できるのか，とかをちゃんとは理解できていない...）

<br>

## 5.3 暗黙的な獲得

「行動から嗜好を予測するのは難しい」という根本的な問題に対して，いろいろな工夫がなされている．

> 例えば，マイクで収集した発話内容 [高間 07] や，アイカメラを使って求めた注視領域 [吉高 07] などを利用する試みなどである．

このように画像認識,音声認識,自然言語処理などを使う方法もある．


<br>

## 5.4 嗜好データのその他の要因

> 利用者が初めてシステムを利用するときに，特定のアイテム群について明示的に質問して，嗜好データを集めることが考えられる．全ての利用者が共通に評価しているアイテム群があると，8 章の協調フィルタリングでは利用者間の嗜好の類似性を評価しやすくなる利点がある．

[MovieLens](https://movielens.org/home)を始めた時に下図のような質問をされたが，こういう意味があったんだな．

![4A1A6DAE-51DF-44A6-B7FF-73DB96D6DAC6_1_105_c](https://user-images.githubusercontent.com/55879719/79245816-90189f80-7eb3-11ea-81e0-fea0ad561c2c.jpeg)




<br>
<br>

# 第6章 嗜好の予測

<br>

ここでようやく機械学習・統計モデルの出番．O-I-P の一部分でしかない．

> 嗜好の予測段階の実現方法は大きく二つに分類される．レンタルビデオ店で，顧客が見たい映画を推薦する場合を考えてみよう．一つは，ファンである監督，好みのジャンルを利用者に尋ねてその条件に合ったものを選ぶ方法である．これを，検索対象の内容を考慮して推薦をするので**内容ベースフィルタリング** (content-based filtering)と呼ぶ．もう一つは，映画の趣味が似ている知り合いに，面白かった映画を教えてもらう「口コミ」の過程を自動化する方法である．他の人との協調的な作業によって推薦対象を決めるため，この推薦手法は**協調フィルタリング** (collaborative filtering)や社会的フィルタリング (social filtering) と呼ばれている．

ここがわかりやすい．

内容ベースフィルタリングの手順は...

1. 各アイテムの特徴ベクトルを用意しておく．例えばラーメンなら (スープの種類, 麺の太さ, 価格) というベクトル．
2. アクティブユーザの利用者プロファイルを直接質問あるいは行動解析で取得する．例えば (好きなスープ, 好きな麺の太さ, 好きな価格) というベクトル．
3. アイテム特徴ベクトルと利用者プロファイルベクトルの類似度を見て，好みに合うアイテムをレコメンド

協調フィルタリングの手順(イメージ)は...

1. アクティブユーザと嗜好が似ている(あるいは真逆な)ユーザを探す．Aさんとは同じだ，Bさんとは関連なさそう，Cさんとは真逆だ，みたいな．
2. それを考慮しながら，各ユーザがアイテムに与えた評価を参照する．Aさんが好きって言ってるから俺も好きだろう，Cさんが嫌いって言ってるから俺は好きだろう，みたいな．

<br>

## 6.1　内容ベースと協調フィルタリングの比較


> 協調フィルタリングと内容ベースフィルタリングの長所と短所を [Balabanovic 97,
Burke 02] などに基づき表 6.1 にまとめた

この表6.1めっちゃわかりやすいが，ちょっと加筆して作り直す↓



|    |協調|内容ベース|
|----|----|----|
|セレンディピティ|有利|不利|
|ドメイン知識|不要|必要!!|
|複数ドメインのアイテム|扱える|扱えない|
|スタートアップ問題|問題あり|基本大丈夫|
|利用者数|多数必要|1人でも可|
|coverage|不人気or新商品は無理|基本100%|
|類似アイテム|色違い等の扱い面倒|扱いやすい|
|少数派の利用者|レコメンド難しい|難しくない|

これらの比較項目について，１つずつ掘り下げていく．

<br>

### 6.1.1 多様性

内容ベースの場合，利用者プロファイル(ユーザの嗜好パターン)とアイテム特徴ベクトルを直接比較するので，過去に選んだものと似たような物ばっかレコメンドすることになる．セレンディピティ低い．

協調フィルタリングの場合「嗜好がだいたい似ているけど個人差レベルの違いはある」サンプルユーザを参考にし，その人が好きなアイテムをレコメンドする．
なんというか，アクティブユーザの嗜好パターンを直接アイテムと比較するんじゃなく，他人の嗜好を間に挟んでいる．よって，まだ見たことない意外な商品がレコメンドされたりして，セレンディピティが高い．

<br>

### 6.1.2 ドメイン知識

協調フィルタリングではアイテムの特徴(スープの味,麺の太さ)は全く使わない．

一方，内容ベースフィルタリングでは事前にアイテム特徴ベクトルを作っておく必要がある．
これを作るにはドメイン知識が必須で，相当大変な作業．具体的には...

- アイテムのデータを集める手間やコストがでかいし，その方法もドメインによって違う．ラーメンのデータなんてまとまっているはずない．

- ユーザの嗜好・アイテムの性質を十分に表現できるような特徴を入れないと，レコメンドがうまくいかない．ラーメンでスープと麺以外の特徴って素人には思いつかない．

また，近い話として

> 例えば，ある映画が好きでも，内容ベース法では，そのサントラの CD を推薦することは難しい．なぜなら，映画と CD は，異なる特徴ベクトルで表現されているためである．

も面白い．逆にいうと協調フィルタリングではこれが可能．

<br>

### 6.1.3 スタートアップ問題

コールドスタート問題とも呼ばれ，

- 新規ユーザに適切なレコメンドをする難しさ
- 新規アイテムをレコメンドする難しさ

の2つに分けられる．

協調フィルタリングの場合，新規ユーザも新規アイテムも無理．ユーザからアイテムへの評価値がベースだから．

内容ベースフィルタリングの場合，新規アイテムのレコメンドは問題なく可能．登録されたアイテムの特徴さえあればOKなので．新規ユーザへのレコメンドは利用者プロファイルを初回質問などで取得できれば問題なく可能．

以上のことから，

> 観光地の案内端末での推薦など，同一利用者の継続的な利用があまりない状況では，直接指定型の内容ベース法を採用すべきである．

> 商品が頻繁に入れ替わるような場合は，内容ベースフィルタリングが有利である．

のように状況に応じた使い分けをすべき．

<br>

### 6.1.4 利用者数

> 内容ベース法の場合は，たとえシステムの利用者が一人であっても推薦は可能である．一方，協調フィルタリングは，他の利用者の意見を参照するので，利用者数がある程度なければ実行できない．

確かに．

<br>

### 6.1.5 被覆率

これはコールドスタート問題で触れたが，新規アイテムのレコメンドは協調だと無理だが内容ベースだと可能．

内容ベースでは，アイテム特徴ベクトルに欠損がない限り coverage 100%．

<br>

### 6.1.6 類似アイテム

協調フィルタリングではアイテムの特徴を一切考慮しないので，

> 推薦されたアイテムを利用者が拒否した場合，それのサイズや色が違うだけの類似アイテムを推薦されてしまう場合も，協調フィルタリングでは生じる．例えば，ある商品を却下したすぐ後で，その商品の色違いを推薦される場合はよく生じうる．

という問題点がある．

「色違いは同じアイテムと見なす」とすれば回避可能だが，

> どのアイテムを同じとみなすかは，アイテムのドメイン依存した難しい問題である．例えば，服飾などでは色の違いは重視されるだろうが，ティッシュペーパなどは色違いでも同じアイテムとみなして良いだろう

という感じで面倒．

<br>

### 6.1.7 少数派の利用者

> 例えば，ほとんど無名なタレントだが，利用者と出身地が同じであるのでファンである人がいたとしよう．こうした嗜好を持つ人は非常に希であろう．さらに，そうした人が同じシステムを利用していることはさらに希である．すると，協調フィルタリングでは，類似した嗜好の人がいなければ嗜好を予測できないので，こうした観点からの推薦は難しい．一方，内容ベースフィルタリングでは，タレントの出身地情報を用いて適切な推薦をすることも可能である．

確かに．

全体的に，ドメイン知識を用いた事前準備が大変な分，内容ベースのがきめ細かいレコメンドができそう．






<br>
<br>

# 第7章 推薦の提示

<br>

モデル(アルゴリズム)での予測結果をもとに「どういう見せ方で」レコメンドするか．

行動経済学・消費者心理学みたいな分野と関連があるぽい．

今回は詳しく触れないが，7.5節は面白いので少し紹介．

> 利用者は，不必要に高価なものを薦められていると疑ったりするため，推薦したアイテムを必ずしも採用するわけではない．採用されない推薦は無意味なので，推薦ができるだけ採用されるような工夫が必要である．そうした工夫として，アイテムの**推薦理由** (explanation of recommendations) も示すことが有効とされている．

[Amazon](https://www.amazon.co.jp/%E7%8F%BE%E5%A0%B4%E3%81%A7%E4%BD%BF%E3%81%88%E3%82%8B%EF%BC%81PyTorch%E9%96%8B%E7%99%BA%E5%85%A5%E9%96%80-%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E4%BD%9C%E6%88%90%E3%81%A8%E3%82%A2%E3%83%97%E3%83%AA%E3%82%B1%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%B8%E3%81%AE%E5%AE%9F%E8%A3%85-AI-TECHNOLOGY-%E4%B8%96%E6%A9%8B/dp/479815718X/ref=sr_1_4?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&dchild=1&keywords=pytorch&qid=1586284133&sr=8-4)の
「この商品をチェックした人はこんな商品もチェックしています」は，推薦理由を説明できている．（永続的個人化じゃなくて一時的個人化だけど．）

図7.2の例では「あなたと嗜好が似たユーザーはこの商品に高評価を付けています．だから推薦しました．」みたいな推薦理由の説明もある．







<br>
<br>

# 第８章 協調フィルタリング

<br>

> 協調フィルタリングによる嗜好の予測について述べる．ここでいう予測とは，活動利用者(active user)はまだ知らないが，他の標本利用者(sample user)は知っているアイテムについて，活動利用者の関心の有無や，評価値を推定することである．こ

- active user：今アイテムをレコメンドしてあげようとしてるユーザー
- sample user：過去に購買して嗜好データがDBに残っておりレコメンドの参考になるユーザーたち


> 利用者数やアイテム数などのデータの特性，望ましい推薦が備えるべき性質を考慮してモデルやアルゴリズムを選択しなくてはならない．もしこれらが適切でなければ，データがいくらあっても予測精度が向上することはない

これがレコメンドシステムの方法論を理解・研究する意義．


<br>

## 8.1 メモリベース法とモデルベース法


> すなわち，事前にモデルを構築するかどうかという違いが重要である．

が分かりやすい．協調フィルタリングは，次の2つに分けられる．
- モデルベース法：モデル構築は事前に済ませておき，レコメンド実行の際はそのモデルでの予測値算出を行う．
- メモリベース法：レコメンド実行の度にモデル構築と予測値算出の両方を行う．


表8.1が分かりやすい．

推薦時間について．
毎回のレコメンドの際にモデル構築をやり直す「メモリベース法」は「モデルベース法」に比べて遅い．

適応性について．
ユーザやアイテムの追加削除があって分布構造が変化した時に
「メモリベース法」では毎回最新の情報を使ってモデル構築するので問題なく対応できるが，
「モデルベース法」ではモデルを作り直さないと対応できず不適切なレコメンドがされてしまう場合がある．
例えば昔からずっとモデルを更新していないと「坂道アイドル好き」のような最近よくある嗜好パターンを補足できない．

この問題について，

> モデルベース法では，夜間などシステムが利用されない間や，毎月や毎週など定期的にモデルをあらかじめ構築・更新しておく．

という対策が取られるが，これをどのくらいの頻度で行うかは運用・ビジネス次第．話し合いが必要．


先に述べておくと，アルゴリズムを基準にすると次のように分類できる．

- 内容ベースフィルタリング(利用者プロファイルとアイテム特徴ベクトルの直接比較でレコメンド)
- 協調フィルタリング(類似ユーザの嗜好を参考にレコメンド)
  - メモリベース協調フィルタリング(レコメンドの度にモデル構築・予測値算出)
    - 利用者間メモリベース法 (9.1節)
    - アイテム間メモリベース法 (9.2節)
  - モデルベース協調フィルタリング(事前にモデル構築してレコメンドの時に予測値算出)









<br>
<br>

# 第９章 メモリベース型協調フィルタリング

<br>

> 利用者間型は，活動利用者と嗜好パターンが似ている利用者をまず見つけ，彼らが好むものを推薦する．一方，アイテム間型では，活動利用者が好むアイテムと類似したアイテムを推薦する．

というように別れるが，分かりやすいので一旦前者をベースに説明していく．

> 機械学習の観点からは k 近傍法 (k-nearest neighbor method) とみなせる

これについては後述の数式を見るとわかる．
1人のユーザーの嗜好(アイテム評価)を1本のベクトルで表現して，その類似度(ベクトル間の距離とか)を考慮して考えるから．


<br>

## 9.1 利用者間型メモリーベース法

利用者間型メモリベース協調フィルタリングの代表的な「手法」として，GroupLens てのがある．
で，これは，[MovieLens](https://movielens.org/home) っていう「サービス」に実装されている．


> レストランを探す場合のことを考えてみよう．
このとき，自分と食べ物の嗜好が似ている何人かの人に尋ねてみて，
彼らの意見をもとにどの店で食事をするか決めたりすることがあるだろう．

GroupLens のやってることは本当にこれに尽きる．より厳密には，
自分と嗜好が真逆なユーザーの意見も参考にする．このアイテム，Aさんが好きって言うなら俺は嫌いだわきっと．

<br>

### 文字や用語の設定

> 評価値行列 $R$ は利用者 $x \in X$ のアイテム $y \in Y$ への評価値 $r_{xy}$ を要素とする行列

> 5 段階のスコアを用いた採点法（5.2 節を参照）で獲得した嗜好データであれば，評価値の定義域は $\mathcal{R} = \{1,\ldots,5\}$

のとこについて．

$\mathcal{X} = \{1,\ldots,n\}$ は $n$人のユーザーの集合，$\mathcal{Y} = \{1,\ldots,m\}$ は $m$個のアイテムの集合として定義されているが，評価値 $r_{xy}$ は厳密には定義されていない気がする．

「明示的or暗黙的に取得した，ユーザ x がアイテム y を好む度合いを表していると思われる量的尺度の値」という感じか．
例えば...
- 明示的に獲得した評価値：MovieLens みたいに利用者がつけた rate の値(星の数)
- 暗黙的に獲得した評価値：クリック回数，そのページを眺めていた時間

他のノーテーションは本文参照．

<br>

### 類似度の計算(9.1式)

> 類似度とは，嗜好パターンがどれくらい似ているかを定量化

という定義なので，$R$ の行ベクトルの似ている度合いを測れれば良く，方法は色々ある．

パッと思いつくのはユークリッド距離だが，これは微妙．
なぜなら「大きさが違ってもベクトルの方向が同じなら似ている思考パターンと見なす」と考える方が自然だから．

なので，相関係数(9.1式)やコサイン類似度が使われる．

ただいずれにせよ「共通に評価しているアイテム」しか使えないという問題があり，たとえば，

> 共通に評価したアイテムが一つ以下ならば，Pearson 相関は計算できないので 0 とする．

は妥当性ない．共通アイテム数を考慮した改良もあるが一旦触れない．

<br>

### 嗜好の予測(9.2式)

(9.2)式の第2項は，
$$
\begin{aligned}
& \frac{\sum_{x \in \mathcal{X_y}} \rho_{ax}(r_{xy}-\bar{r}_x')}{\sum_{x \in \mathcal{X}_y}|\rho_{ax}|} \\
&= \sum_{x \in \mathcal{X}_y} \left\{
\frac{|\rho_{ax}|}{\sum_{x \in \mathcal{X}_y} |\rho_{ax}|}
\mathrm{sign}(\rho_{ax})(r_{xy}-\bar{r}_x')
\right\}
\end{aligned}
$$
って感じで式変形するとわかりやすい．具体的には...

$(r_{xy}-\bar{r}_x')$の部分は，$x$ さんが アイテム $y$ を
「他のアイテムと比べて(自分の付けた評価値の平均値と比べて)どのくらい好きか/嫌いか」の値．つまり，個人によって良い rate つけまくる人とか逆の人とかいるけど，調整を行ってそういうノイズを除去した上での$x$ さんのアイテム $y$ に対する評価値．

$\mathrm{sign}(\rho_{ax})$ の部分は，
アクティブユーザ $a$ さんと
嗜好(の方向)が同じ人の意見はそのまま受け取り，
嗜好(の方向)が逆の人の意見は逆にして受け取る，という役割．
嗜好が同じ人の「うーん平均より好きかな/嫌いかな」はそのまま参考にし，
逆の人の「うーん平均より好きかな/嫌いかな」は逆にして受け取る．

$\frac{|\rho_{ax}|}{\sum_{x \in \mathcal{X}_y} |\rho_{ax}|}$
の分子は「嗜好パターンに強い関連がある人(嗜好がめっちゃ同じあるいはめっちゃ逆な人)の意見は重視したいので大きい重みを課す」という意味．分母は「分子が極端な値とならないようにする，具体的には $(r_{xy}-\bar{r}_x')$ の最小値以上最大値以下の範囲に収まるようにする」ためだと思う．

上記3つをまとめると，(9.2)式の第2項は，サンプルユーザーが「自分と嗜好が逆なのか同じなのか」と「自分の嗜好パターンにどのくらい関連があるのか」を考慮した上で，サンプルユーザたちの「平均より好きかな/嫌いかな」をまとめたもの．
それを第1項の「自分がつけた評価値の平均値」に足し合わせて，それを「未評価のアイテム $y$ への評価値の予測値」としている．

より分かりやすく言うと...
1. 全ユーザに対して「アイテムy(固定)はあなたの中で(あなたの平均と比べて)どのくらい好き?」と聞く．
2. (ユーザと自分の類似度を考慮した上で)それを混ぜ合わせて「このアイテムは私の中で(私の平均と比べて)このくらい好きだろう」というのを求める．
3. それを「私の平均」に足すことで「このアイテム私はこのくらい好き」を推定する．

<br>

### 9.1.1 利用者間型メモリベース法の改良

> 利用者の近傍を使う改良もある．ここでの近傍とは，式 (9.1) の相関が大きな，すなわち活動利用者と類似した嗜好をもつ利用者の集合のことである．式 (9.2) の推定評価値は，アイテム y を評価済みの全ての利用者の評価に基づいているが，これを事前に計算した活動利用者の近傍のみに基づいて計算する．実験によれば，近傍利用者数がある程度以上になると，それ以降は増やしても予測精度は向上しない．よって，近傍利用者だけに計算を限定することで計算量を減らし，データベースの参照も抑制できるので，効率よく計算できるようになる．

これは k 近傍法に近い考え方で，確かに計算コストは下がるが，

> ただし，モデルベース法のモデルほど頻繁にする必要はないが，近傍は定期的に更新しなくてはならないので，純粋なメモリベース法の利点は部分的には失われる．また，新規の参加者については近傍を新たに計算する必要が生じ，脱退者が他の利用者の近傍利用者であれば予測精度の低下を招く．

という問題がある．それはそう．毎回のレコメンドの度に近傍計算しなおせば問題無いが計算がきついので，どのくらいメモリベースの利点を手放すのか考える必要あり．

また，**基本的に評価値行列 $R$ は欠損値だらけ**であり，それに対処するための改良もたくさんある．

欠損のない(共通して評価されている)アイテムのみを使うより，
補完してでも全てのアイテムを使った方が精度が上がる時もある．

- デフォルト投票(平均値補完)
- EMアルゴリズムによる補完

などが挙げられている．


<br>

## 9.2 アイテム間型メモリベース法

ユーザー $a$ のアイテム $y$ に対する嗜好を求めたいとすると...

- ユーザー間型：(評価値行列 $R$ の行ベクトルの類似度をもとに) ユーザー $a$ と嗜好が似ているユーザーを探し，その人がアイテム $y$ を好きかどうかを参考にする．
- アイテム間型：(評価値行列 $R$ の列ベクトルの類似度をもとに) アイテム $y$ とファン層が似ているアイテムを探し，それを ユーザー $a$ が好きかどうかを参考にする．

という感じなので，方法論は大体同じ．省略する．

別の角度から言うと，

- ユーザー間型：アクティブユーザーと嗜好が似ている他のユーザーが好きなアイテムをレコメンドする．
- アイテム間型：アクティブユーザーが好きなアイテムとファン層が似ているアイテムをレコメンドする.

となる．

ここで「アイテムのページを閲覧する＝そのアイテムが好きである」と仮定すると，アイテム間型の考え方を使って次の手順で一時的個人化のレコメンドをすることができる．

1. ユーザー $a$ がアイテム $z$ のページを開いたとする．
2. アイテム $z$ とファン層(評価値行列 $R$ の列ベクトル)が似ている他のアイテムを探す．
3. それをレコメンドする．



<br>
<br>
<br>


----

<br>
<br>
<br>


10.2.1の
> U の 第 y 列ベクトル u_y はアイテム y の特徴を表している．

は恐らく誤字で，正しくは「V の第 y 列ベクトル v_y はアイテム y の特徴を表している」だと思う．根拠は紙メモ参照．ブロック行列でちゃんと意味を考えればわかる．