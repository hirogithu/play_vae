# ファイル構成

- "main.py" : face recommender system. 入力画像に類似した画像を検索表示する．
- "make_search_space.py" : VAEの学習用．重みを保存する．
- "vae_sample.py" : VAEの性能把握用．学習済みのVAEに指定の画像を入力し，出力を得る．
- utils/ : main.pyのモジュール．
- vae/ : VAEのモデル構成
- results/ :