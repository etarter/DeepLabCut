import deeplabcut
import tkinter
from tkinter import filedialog
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import pandas as pd

def file_dialog(type):
    root = tkinter.Tk()
    root.geometry('0x0+0+0')
    root.lift()
    root.focus_force()
    root.deiconify()
    root.update_idletasks()
    #root.overrideredirect(True)
    #root.mainloop()
    if type == 'file':
        path = filedialog.askopenfilename(parent=root, initialdir= os.getcwd(), title= "select config file")
    elif type == 'file3d':
        path = filedialog.askopenfilename(parent=root, initialdir= os.getcwd(), title= "select config3d file")
    elif type == 'directory':
        path = filedialog.askdirectory(parent=root, initialdir=os.getcwd(), title='select video directory')     #add automatic video directory assignement
    elif type == 'vfile':
        path = filedialog.askopenfilename(parent=root, initialdir=os.getcwd(), title='select video file')
    root.withdraw()
    return path

def new_project():
    print('(y/n/exit) do you want to create a new project?')
    console = input()
    loop = True

    while loop:
        if (console == 'y') or (console == 'yes'):
            print('project name')
            project_name = input()
            print('your name')
            your_name = input()
            print('enter videos directory')
            #videos_dir = os.path.join(file_dialog('directory'))
            videos_dir = r'C:\Users\etarter\Downloads\videos'
            #config_path = deeplabcut.create_new_project(project_name, your_name, [videos_dir], videotype='.mp4', copy_videos=False, multianimal=True)
            config_path = r'C:\Users\etarter\Downloads\dlc\full3dma-dlc-2020-11-23\config.yaml'
            #config_path3d = deeplabcut.create_new_project_3d(project_name, your_name, num_cameras=2)
            config_path3d = r'C:\Users\etarter\Downloads\dlc\full3dma-dlc-2020-11-23-3d\config.yaml'
            print('edit config files accordingly')
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_path])
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_path3d])
            loop = False
        elif (console == 'n') or (console == 'no'):
            config_path = r'C:\Users\etarter\Downloads\dlc\hybrid-dlc-2020-11-23\config.yaml'#os.path.join(file_dialog('file'))
            config_path3d = r'C:\Users\etarter\Downloads\dlc\full3dma-dlc-2020-11-23-3d\config.yaml'#r'C:\Users\etarter\Downloads\dlc\hybrid-dlc-2020-11-23-3d\config.yaml'#os.path.join(file_dialog('file3d'))
            videos_dir = r'C:\Users\etarter\Downloads\videos'#os.path.join(file_dialog('directory'))
            loop = False
        elif console == 'exit':
            loop = False
        else:
            print('try again')
            console = input()
    return config_path, config_path3d, videos_dir

config_path, config_path3d, videos_dir = new_project()

def run_workflow(config_path, config_path3d, videos_dir):
    loop = True
    what_to_do = '(e) extract frames\n(l) label frames\n(t) create, train and evaluate network \n(a) analyze videos\n(c) calibrate and triangulate\n(r) extract outliers and refine labels\n(m) merge and retrain network\n(v) add new videos\n(exit) exit'
    print(what_to_do)
    answer = input()
    while loop:
        if answer == 'e':
            deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, cluster_resizewidth=10, cluster_step=1)
            print(what_to_do)
            answer = input()
        elif answer == 'l':
            deeplabcut.label_frames(config_path)
            print(what_to_do)
            answer = input()
        elif answer == 't':
            deeplabcut.create_multianimaltraining_dataset(config_path, net_type='resnet_50')
            print('max iterations')
            iterations = input()
            deeplabcut.train_network(config_path, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0)
            deeplabcut.evaluate_network(config_path, gputouse=0, plotting=True)
            deeplabcut.evaluate_multianimal_crossvalidate(config_path, plotting=True)
            print(what_to_do)
            answer = input()
        elif answer == 'a':
            scorername = deeplabcut.analyze_videos(config_path, [videos_dir], videotype='.mp4', gputouse=0, save_as_csv=False)
            deeplabcut.convert_detections2tracklets(config_path, [videos_dir], videotype='.mp4', track_method='box')
            videos = range(len(glob.glob(videos_dir+'\*_bx.pickle')))
            for video in videos:
                man, viz = deeplabcut.refine_tracklets(config_path, os.path.join(videos_dir, glob.glob(videos_dir+'\*_bx.pickle')[video]), os.path.join(file_dialog('vfile')))
            print(what_to_do)
            answer = input()
        elif answer == 'c':
            cams = range(len(glob.glob(videos_dir+'\*_bx.h5')))
            for cam in cams:
                ma_h5 = pd.read_hdf(glob.glob(videos_dir+'\*_bx.h5')[cam])
                sa_h5 = pd.read_hdf(glob.glob(videos_dir+'\hybrid\*_10000.h5')[cam])
                outfile = os.path.join(videos_dir, os.path.basename(glob.glob(videos_dir+'\hybrid\*_10000.h5')[cam]))
                ma_h5.columns = sa_h5.columns
                ma_h5.to_hdf(outfile, key="df_with_missing", mode="w")
            #calibration and triangulation
        elif answer == 'r':
            print()
        elif answer == 'm':
            print('max iterations')
            iterations = input()
            deeplabcut.merge_datasets(config_path)
            deeplabcut.create_multianimaltraining_dataset(config_path, net_type='resnet_50')
            deeplabcut.train_network(config_path, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0)
            deeplabcut.evaluate_network(config_path, gputouse=0, plotting=True)
            deeplabcut.evaluate_multianimal_crossvalidate(config_path, plotting=True)
            print(what_to_do)
            answer = input()
        elif answer == 'v':
            new_videos_dir = os.path.join(file_dialog('directory'))
            deeplabcut.add_new_videos(config_path, [new_videos_dir], copy_videos=False)
            print(what_to_do)
            answer = input()
        elif answer == 'exit':
            loop = False
        else:
            print('try again')
            answer = input()

run_workflow(config_path, config_path3d, videos_dir)
