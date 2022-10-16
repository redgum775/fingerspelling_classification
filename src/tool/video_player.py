import os
import cv2
import datetime
import pandas as pd
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from PIL import Image, ImageTk, ImageOps

def create_dir(export_dir):
  dirs = os.listdir(export_dir)
  if len(dirs) == 0:
    number = '1'.zfill(8)
  else:
    number = str((max(0, int(max(dirs))) + 1)).zfill(8)
  os.makedirs(f'{export_dir}/{number}', exist_ok=True)
  dir_path = f'{export_dir}/{number}'
  return dir_path

class ClipVideo():
  def __init__(self, origin_video):
    self.origin_video = origin_video
    # encoder(for mp4)
    self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    self.width = self.origin_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.height = self.origin_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.fps = self.origin_video.get(cv2.CAP_PROP_FPS)
    print(f'INFO: Origin Video performance is [width: {self.width}, height: {self.height}, fps: {self.fps}].')

  def clip(self, label, from_=0, to=0):
    dir_path = create_dir('./datasets/end-to-end/ja')
    video_path = f'{dir_path}/video.mp4'
    self.clip_video = cv2.VideoWriter(video_path, self.fourcc, int(self.fps), (int(self.width), int(self.height)))
    self.origin_video.set(cv2.CAP_PROP_POS_FRAMES, int(from_))
    while(True):
      ret, frame = self.origin_video.read()
      if ret is False:
        break
      if int(self.origin_video.get(cv2.CAP_PROP_POS_FRAMES) > int(to)):
        print('INFO: Video Cliped')
        break
      self.clip_video.write(frame)
    self.clip_video.release()
    
    df = pd.read_csv('./datasets/end-to-end/data_list.csv')
    id = df['id'].values[-1]+1
    new_data = pd.DataFrame(
      [[id, label, datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), dir_path, video_path]], 
        columns=['id', 'label', 'date', 'data_dir_path', 'base_video_file_path']
    )
    df = pd.concat([df, new_data], axis=0)
    df.to_csv('./datasets/end-to-end/data_list.csv', index=False)

class VideoPlayer(tk.Frame):
  def __init__(self, master=None):
    super().__init__(master)
    self.config(width=640, height=550)

    self.fps = 30
    self.video = None
    self.frame = None
    self.frame_num = 0
    self.frame_length = 0

    self.play_speed = 1.0
    self.play_flg = False

    self.create_main_content(master)
    self.create_side_content(master)

    self.video_frame_timer()

  def create_main_content(self, master=None):
    main_content = tk.Frame(master, width=640, height=600, bg='#eeeeee')
    label = tk.Label(main_content, text='VideoPlayer', bg='#afafaf')
    label.pack(fill='x')
    self.create_video_canvas(main_content)
    self.create_ctrl_panel(main_content)
    main_content.pack(side=tk.LEFT, anchor=tk.NW, padx=5, pady=5)

  def create_video_canvas(self, master=None):
    self.video_canvas = tk.Canvas(master, width=640, height=480, bg='#000000')
    self.video_canvas.pack(padx=5, pady=5)
  
  def create_ctrl_panel(self, master=None):
    ctrl_panel = tk.Frame(master, width=640, height=50)
    self.create_play_button(ctrl_panel)
    self.create_change_speed_button(ctrl_panel)
    self.create_seek_bar(ctrl_panel)
    ctrl_panel.pack(padx=30)
  
  def create_seek_bar(self, master=None):
    self.scale_var = tk.DoubleVar()
    self.seek_bar = tk.Scale(
      master,
      variable = self.scale_var,
      command = self.slider_scroll,
      orient = tk.HORIZONTAL,
      length = 600,
      width = 20,
      sliderlength = 20,
      from_ = self.frame_num,
      to = self.frame_length,
      tickinterval = int(self.frame_length/5),
      troughcolor='#0000ff'
    )
    self.seek_bar.pack(side=tk.LEFT)
  
  def slider_scroll(self, event=None):
    self.play_flg = False
    self.play_button.config(text='▶')
    pos = self.scale_var.get()
    self.video.set(cv2.CAP_PROP_POS_FRAMES, pos)
    self.show_img()
  
  def create_play_button(self, master=None):
    self.play_button = tk.Button(master, text="▶", width=2, command=self.state_change)
    self.play_button.pack(side=tk.LEFT, padx=5)
  
  def state_change(self):
    if self.video is None:
      messagebox.showerror(title='Error', message='ビデオが設定されていません')
      return
    self.play_flg = not self.play_flg
    if self.play_flg:
      self.play_button.config(text='ll')
    else:
      self.play_button.config(text='▶')
  
  def create_change_speed_button(self, master=None):
    self.change_speed_button = tk.Button(master, text="x1.0", width=2, command=self.click_change_speed)
    self.change_speed_button.pack(side=tk.LEFT, padx=5)

  def click_change_speed(self):
    if self.play_speed == 1.0:
      self.play_speed = 1.5
      self.change_speed_button.config(text='x1.5')
    elif self.play_speed == 1.5:
      self.play_speed = 2.0
      self.change_speed_button.config(text='x2.0')
    elif self.play_speed == 2.0:
      self.play_speed = 3.0
      self.change_speed_button.config(text='x3.0')
    elif self.play_speed == 3.0:
      self.play_speed = 5.0
      self.change_speed_button.config(text='x5.0')
    elif self.play_speed == 5.0:
      self.play_speed = 1.0
      self.change_speed_button.config(text='x1.0')

  def create_side_content(self, master=None):
    side_content = tk.Frame(master, width=550, height=600, bg='#eeeeee')
    self.create_config_panel(side_content)
    self.create_video_clip_panel(side_content)
    side_content.pack(side=tk.LEFT, anchor=tk.NW, padx=5, pady=5)

  def create_config_panel(self, master=None):
    config_frame = tk.Frame(master, bg='#eeeeee')
    label = tk.Label(master, text='Config', bg='#afafaf')
    label.pack(fill='x', padx=5, pady=5)
    self.path_frame = tk.Frame(config_frame, bg='#eeeeee')
    label = tk.Label(self.path_frame, text='動画ファイル', bg='#eeeeee')
    label.pack(side=tk.LEFT)
    self.path_entry = tk.Entry(self.path_frame, width=25)
    self.path_entry.pack(side=tk.LEFT)
    self.ref_button = tk.Button(self.path_frame, text='参照', command=self.open_filedialog)
    self.ref_button.pack(side=tk.LEFT, padx=5)
    self.close_button = tk.Button(self.path_frame, text='閉じる', command=self.close_video)
    self.close_button.pack(side=tk.LEFT)
    self.path_frame.pack(side=tk.LEFT, padx=5, pady=5)
    config_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
  
  def open_filedialog(self):
    file = filedialog.askopenfilename(filetypes = [(['video',['*.mp4', '*.MP4']])])
    self.path_entry.delete(0, tk.END)
    self.path_entry.insert(0, file)
    self.get_video(self.path_entry.get())
  
  def close_video(self):
    if self.video is not None:
      self.video.release()
      self.video = None
    self.play_flg = False
    self.path_entry.delete(0, tk.END)
    self.seek_bar.config(to=0, tickinterval=0)
  
  def create_video_clip_panel(self, master=None):
    video_catting_panel = tk.Frame(master, bg='#eeeeee')
    label = tk.Label(video_catting_panel, text='video clipping', bg='#afafaf')
    label.pack(fill='x')
    clip_range = tk.Frame(video_catting_panel)
    label = tk.Label(clip_range, text='from', bg='#eeeeee')
    label.pack(side=tk.LEFT)
    self.clip_from = tk.Entry(clip_range, width=10)
    self.clip_from.pack(side=tk.LEFT)
    label = tk.Label(clip_range, text='to', bg='#eeeeee')
    label.pack(side=tk.LEFT)
    self.clip_to = tk.Entry(clip_range, width=10)
    self.clip_to.pack(side=tk.LEFT)
    clip_range.pack(side=tk.TOP, fill='x', padx=5, pady=5)
    set_label = tk.Frame(video_catting_panel)
    label = tk.Label(set_label, text='label', bg='#eeeeee')
    label.pack(side=tk.LEFT)
    self.clip_video_label = tk.Entry(set_label)
    self.clip_video_label.pack(side=tk.LEFT)
    set_label.pack(side=tk.TOP, fill='x', padx=5, pady=5)
    self.clip_button = tk.Button(video_catting_panel, text='clip', command=self.click_clip_button)
    self.clip_button.pack(side=tk.TOP, fill='x', padx=5, pady=5)
    video_catting_panel.pack(side=tk.TOP, fill='x', padx=5, pady=5)

  def click_clip_button(self):
    clip_video = ClipVideo(self.video)
    clip_video.clip(self.clip_video_label.get(), self.clip_from.get(), self.clip_to.get())
    self.clip_from.delete(0, tk.END)
    self.clip_to.delete(0, tk.END)
    self.clip_video_label.delete(0, tk.END)

  def get_video(self, video_source=0):
    if self.video is not None:
      self.video.release()
    self.play_flg = False
    self.video = cv2.VideoCapture(video_source)
    self.fps = self.video.get(cv2.CAP_PROP_FPS)
    self.frame = None
    self.frame_num = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
    self.frame_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    self.seek_bar.config(to=self.frame_length, tickinterval = int(self.frame_length/5))
  
  def next_frame(self):
    if not self.play_flg:
      return
    self.frame_num = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
    self.scale_var.set(self.frame_num)
    self.show_img()

  def show_img(self):
    ret, self.frame = self.video.read()
    if ret is False:
      self.play_flg = False
      self.play_button.config(text='▶')
      return
    rgb_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    pad_img = ImageOps.pad(pil_img, (640, 480))
    self.frame = ImageTk.PhotoImage(pad_img)
    self.video_canvas.create_image(640/2, 480/2,image=self.frame)
  
  def video_frame_timer(self):
    self.next_frame()
    self.master.after(int((1000/self.fps)/self.play_speed), self.video_frame_timer)

if __name__ == '__main__':
  root = tk.Tk()
  root.title('VideoPlayer')
  root.config(bg='#cccccc')
  root.geometry('1080x600')
  video_player = VideoPlayer(master=root)
  root.mainloop()