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

## train_using_ctcloss.ipynb  
CTC-Lossを利用した指文字分類モデルの訓練  

## Other
- `training_video_creatin.py` 指文字動画を撮影・保存＆指文字動画ファイルと正解ラベルと対応付けて`datasets\end-to-end\data_list.csv`に保存
- `build_training_dataset.py` 指文字動画をMediaPipeの処理にかけた結果を保存＆保存先を`datasets\end-to-end\data_list.csv`に記入
- `./tool/video_player.py` 動画から任意の場面だけクリッピング＆正解ラベルと対応付け
  
## Reference  
- [Mediapipe](https://github.com/google/Mediapipe)  
- [Keras 2 : examples : コンピュータビジョン – 手書きテキスト (可変長文字列) 認識](https://tensorflow.classcat.com/2021/11/20/keras-2-examples-vision-handwriting-recognition/)

## Author  
Redgum  

## License  
Fingerspelling Classification is under [Apache v2 license](LICENSE).  