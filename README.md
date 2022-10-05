# Fingerspelling Classification  

## train_using_RandomForest.ipynb  
RandomForestを使用する分類モデル構築  

2020年構築で、2022年に動かしたらJavaコード生成でエラーを吐くようになっていた。  
最新のsklearnに対応していないため、sklearnのバージョンを下げたら使用可能。  
ほかは問題なく動くので，Javaコード生成だけコメントアウトして実行する。
  
### detasets/randomforest  
ランダムフォレスト訓練に使用するデータセット置き場  

- aiueo_detaset  
手の形状を表す真偽値のみのデータセット  
- mediapipe_detaset  
手の形状を表す真偽値＋各指の角度情報など数値情報を含めたデータセット  
- handsign_detaset  
手の形状を表す真偽値＋各指の角度情報など数値情報を含めたデータセット  

## Author 
Redgum  

# Reference  
- [Mediapipe](https://github.com/google/Mediapipe)  

# License  
Hand gesture recognition using Mediapipe is under [Apache v2 license](LICENSE).