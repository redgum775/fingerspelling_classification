import datetime
import cv2
import os
import pandas as pd

from utils import drawing_utils
from utils import CvFpsCalc

import random
import copy

def main():
  cvFpsCalc = CvFpsCalc(buffer_len=10)

  mode = 0
  select = 'Moji'
  if select == 'Moji':
    moji = Moji()
    label = moji.get_moji()
    export_dir = './datasets/end-to-end/ja'
  elif select == 'Number':
    num = Number()
    label = num.get_number()
    export_dir = './datasets/end-to-end/num'

  df = pd.read_csv('./datasets/end-to-end/data_list.csv')
  id = df['id'].values[-1]

  # For webcam input:
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  print(f'INFO: Web camera performance is [width: {width}, height: {height}, fps: {fps}].')

  rec_flg = False
  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # キー処理(ESC：終了) #
    key = cv2.waitKey(1)
    if key == 27:
      break #ESC
    mode = select_mode(key, mode)
    if mode == 1: # k押下時　保存モード
      rec_flg = not rec_flg
      if rec_flg:
        dir_path = create_dir(export_dir)
        video_path = f'{dir_path}/video.mp4'
        video = Video(video_path, fps, int(width), int(height))
        id += 1
        new_data = pd.DataFrame(
          [[id, label, datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), dir_path, video_path]], 
          columns=['id', 'label', 'date', 'data_dir_path', 'base_video_file_path']
        )
        df = pd.concat([df, new_data], axis=0)
      else:
        video.release()
      pass
    elif mode == 0: # n押下時　
      flg = False
    elif mode == 2: # c押下時　学習文字変更
      if select == 'Moji':
        label = moji.get_moji()
      elif select == 'Number':
        label = num.get_number()
    elif mode == -1:  # 押下キーなし
      pass

    image = cv2.flip(image, 1)
    if rec_flg:
      video.write(image)
    
    image = drawing_utils.draw_jp_text(image, label, (15, 65), 50)
    image = cvFpsCalc.dipsplay_fps(image)
    if rec_flg:
      cv2.circle(image, (8, 8), 5,(0, 0, 255), thickness=-1)
    cv2.imshow('MediaPipe Hands', image)

  cap.release()
  df.to_csv('./datasets/end-to-end/data_list.csv', index=False)
  if rec_flg:
    video.release()

def create_dir(export_dir):
  dirs = os.listdir(export_dir)
  if len(dirs) == 0:
    number = '1'.zfill(8)
  else:
    number = str((max(0, int(max(dirs))) + 1)).zfill(8)
  os.makedirs(f'{export_dir}/{number}', exist_ok=True)
  dir_path = f'{export_dir}/{number}'
  return dir_path

def select_mode(key, mode):
  if key == 110:  # n
    mode = 0
  elif key == 107:  # k
    mode = 1
  elif key == 99:   # c
    mode = 2
  else:
    mode = -1
  return mode

class Moji:
  def __init__(self):
    self.indices_char = [
                  'あ','い','う','え','お',
                  'か','き','く','け','こ',
                  'さ','し','す','せ','そ',
                  'た','ち','つ','て','と',
                  'な','に','ぬ','ね','の',
                  'は','ひ','ふ','へ','ほ',
                  'ま','み','む','め','も',
                  'や','ゆ','よ',
                  'ら','り','る','れ','ろ',
                  'わ','を','ん']
    self.clone = copy.copy(self.indices_char)

  def get_moji(self):
    if(len(self.clone) == 0):
      self.clone = copy.copy(self.indices_char)

    if 1 == len(self.clone):
      x = 0
    else:
      x = random.randint(0, len(self.clone)-1)
    return self.clone.pop(x)

class Number:
  def __init__(self):
    self.indices_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    self.clone = copy.copy(self.indices_char)

  def get_number(self):
    if(len(self.clone) == 0):
      self.clone = copy.copy(self.indices_char)

    if 1 == len(self.clone):
      x = 0
    else:
      x = random.randint(0, len(self.clone)-1)
    return self.clone.pop(x)

import re
from collections import Counter
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

def matching(text):
  # 同じ要素が５個以上続いたら１つの要素としてみなす・非指文字記号は削除
  t = ''
  for c in re.split(r'_+', text):
    if c == '': continue
    cc = Counter(c)
    if cc.most_common()[0][1] >= 5:
      t += cc.most_common()[0][0]
  return t

if __name__ == '__main__':
  main()

