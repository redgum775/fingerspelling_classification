import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from utils import hand_utils
from utils import drawing_utils

IS_ALL_UPDATE = False

def main():
  df = pd.read_csv('./datasets/end-to-end/data_list.csv')
  map_colmun_and_index = {}
  for index, colmun in enumerate(df.columns, 0):
    map_colmun_and_index[colmun] = index

  for index, row in enumerate(df.values):
    # something to do
    if IS_ALL_UPDATE or pd.isna(row[map_colmun_and_index['mediapipe_processed_video_file_path']])\
      and pd.isna(row[map_colmun_and_index['mediapipe_result_value_file_path']])\
      and pd.isna(row[map_colmun_and_index['feature_file_ptah']]):
        data_dir_path = row[map_colmun_and_index['data_dir_path']]
        base_video_file_path = row[map_colmun_and_index['base_video_file_path']]
        mediapipe_processed_video_file_path, mediapipe_result_value_file_path , feature_file_ptah = mediapipe_process(data_dir_path, base_video_file_path)
        df.at[index, 'mediapipe_processed_video_file_path'] = mediapipe_processed_video_file_path
        df.at[index, 'mediapipe_result_value_file_path'] = mediapipe_result_value_file_path
        df.at[index, 'feature_file_ptah'] = feature_file_ptah
  df.to_csv('./datasets/end-to-end/data_list.csv', index=False)

def mediapipe_process(data_dir_path, base_video_file_path):
  mp_hands = mp.solutions.hands
  mediapipe_path = f'{data_dir_path}/mediapipe.csv'
  attribute_path = f'{data_dir_path}/attribute.csv'
  attribute_f = open(attribute_path, 'w', encoding='utf_8_sig')
  mediapipe_f = open(mediapipe_path, 'w', encoding='utf_8_sig')

  # 動画の情報を取得
  print(f'Video capture from [{base_video_file_path}].')
  cap = cv2.VideoCapture(base_video_file_path)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  output_video_path = f'{data_dir_path}/result.mp4'
  output_video = Video(output_video_path, fps, width, height)

  frame = 1
  with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
      while(cap.isOpened()):
        ret, image = cap.read()
        if ret is False:
          cap.release()
          output_video.release()
          break
        if frame != 1:
          print(f'\r{frame}/{frame_num}: {data_dir_path}', end="")
        else:
          print(f'{frame}/{frame_num}: {data_dir_path}', end="")
        # BGR画像をRGB画像に変換し，mediapipeの処理を実行
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        annotated_image = image.copy()
        if results.multi_hand_landmarks:
          # 画像上にhand landmarkを描画
          for id, (hand_landmarks, hand_world_landmarks, handedness) \
            in enumerate(zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness)):
            annotated_image = drawing_utils.draw_bounding_rect(annotated_image, hand_landmarks)        
            annotated_image = drawing_utils.draw_landmarks(annotated_image, hand_landmarks)
                      # 取得データを入力
            attribute_f.write(f'{hand_utils.get_explanatory_variables_to_csv(hand_landmarks, hand_world_landmarks, handedness)}\n')
            mediapipe_f.write(f'{hand_utils.landmarks_protobuf_to_csv(hand_landmarks)}, ' \
                            f'{hand_utils.landmarks_protobuf_to_csv(hand_world_landmarks)}, ' \
                            f'{hand_utils.classfication_protobuf_to_csv(handedness)}\n')
        else:
          # 取得データをゼロ埋め
          zero_data = '0, ' * 126 + '0, 0, -1'  # ('0' * (21 * 3)) * 2 + '0' + '0' + '-1' 
          mediapipe_f.write(f'{zero_data}\n')
          attribute_f.write(f'0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\n') # '0' * 20
        output_video.write(annotated_image)
        frame += 1
      cap.release()
      output_video.release()
      print("")
      return output_video_path, mediapipe_path, attribute_path

class Video():
  def __init__(self, file_path, fps=30, width=640, height=480):
    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # output file name, encoder, fps, size(fit to image size)
    dir = os.path.dirname(file_path)
    filename = os.path.splitext(os.path.basename(file_path))[0] + '.mp4'
    if os.path.isfile(os.path.join(dir, filename)):
      os.remove(os.path.join(dir, filename))
    self.video = cv2.VideoWriter(os.path.join(dir, filename),fourcc, fps, (width, height))

  def write(self, image):
    self.video.write(image)

  def release(self):
    self.video.release()


if __name__ in '__main__':
  main()