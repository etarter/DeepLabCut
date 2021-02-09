import deeplabcut
import tkinter
from tkinter import filedialog
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import pandas as pd
from pathlib import Path


def file_dialog(type):
    root = tkinter.Tk()
    root.geometry('0x0+0+0')
    root.lift()
    root.focus_force()
    root.deiconify()
    root.update_idletasks()
    if type == 'file':
        path = filedialog.askopenfilename(parent=root, initialdir= os.getcwd(), title= "config_path")
    elif type == 'file3d':
        path = filedialog.askopenfilename(parent=root, initialdir= os.getcwd(), title= "config_path3d")
    elif type == 'filema':
        path = filedialog.askopenfilename(parent=root, initialdir= os.getcwd(), title= "config_pathma")
    elif type == 'directory':
        path = filedialog.askdirectory(parent=root, initialdir=os.getcwd(), title='videos_dir')
    elif type == 'vfile':
        path = filedialog.askopenfilename(parent=root, initialdir=os.getcwd(), title='video_path')
    root.withdraw()
    return path


def open_project():
    print('(n) new project\n(l) load existing')
    console = input()
    loop = True

    while loop:
        if console == 'n':
            print('project_name')
            project_name = input()
            print('your_name')
            your_name = input()
            print('videos_dir')
            videos_dir = os.path.join(file_dialog('directory'))
            config_path = deeplabcut.create_new_project('sa-'+project_name, your_name, [videos_dir], videotype='.mp4', copy_videos=False, multianimal=False)
            config_pathma = deeplabcut.create_new_project(project_name, your_name, [videos_dir], videotype='.mp4', copy_videos=False, multianimal=True)
            config_path3d = deeplabcut.create_new_project_3d(project_name, your_name, num_cameras=2)
            print('edit config files')
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_path])
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_pathma])
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_path3d])
            loop = False
        elif console == 'l':
            model = 'obs'
            if model == 'hybrid':
                config_path = r'C:\Users\etarter\Downloads\dlc\hybrid-dlc-2020-11-23\config.yaml'#os.path.join(file_dialog('file'))
                config_path3d = r'C:\Users\etarter\Downloads\dlc\hybrid-dlc-2020-11-23-3d\config.yaml'#os.path.join(file_dialog('file3d'))
                videos_dir = r'C:\Users\etarter\Downloads\videos'
            elif model == 'full3dma':
                # full3dma 2d config file must point to hybrid model config file
                # full3dma 3d config file must have link to hybrid model config file
                config_path = r'C:\Users\etarter\Downloads\dlc\hybrid-dlc-2020-11-23\config.yaml'
                config_pathma = r'C:\Users\etarter\Downloads\dlc\full3dma-dlc-2020-11-23\config.yaml'
                config_path3d = r'C:\Users\etarter\Downloads\dlc\full3dma-dlc-2020-11-23-3d\config.yaml'
                videos_dir = r'C:\Users\etarter\Downloads\videos'
            elif model == 'test':
                # test 2d config file must point to test1 model config file
                # test1 3d config file must have link to test1 model config file
                config_path = r'C:\Users\etarter\Downloads\dlc\test1-dlc-2021-01-04\config.yaml'
                config_pathma = r'C:\Users\etarter\Downloads\dlc\test-dlc-2021-01-04\config.yaml'
                config_path3d = r'C:\Users\etarter\Downloads\dlc\test1-dlc-2021-01-05-3d\config.yaml'
                videos_dir = r'C:\Users\etarter\Downloads\videos'
            elif model == 'obs':
                # test 2d config file must point to test1 model config file
                # test1 3d config file must have link to test1 model config file
                config_path = r'C:\deeplabcut\dlc\sa-observation-pdz-2021-01-13\config.yaml'
                config_pathma = r'C:\deeplabcut\dlc\observation-pdz-2021-01-13\config.yaml'
                config_path3d = r'C:\deeplabcut\dlc\observation-pdz-2021-01-13-3d\config.yaml'
                videos_dir = r'C:\deeplabcut\videos\12-01-21-model-creation'
            elif model == 'ask':
                config_path = os.path.join(file_dialog('file'))
                config_pathma = os.path.join(file_dialog('filema'))
                config_path3d = os.path.join(file_dialog('file3d'))
                videos_dir = os.path.join(file_dialog('directory'))
            loop = False
        else:
            print('try again')
            console = input()
    return config_path, config_pathma, config_path3d, videos_dir


def workflow(config_path, config_pathma, config_path3d, videos_dir):
    loop = True
    what_to_do = '(e) extract frames\n(l) label frames\n(t) create, train and evaluate network \n(a) analyze videos\n(c) triangulate\n(p) create labeled video\n(r) extract outliers and refine labels\n(m) merge and retrain network\n(v) add new videos\n(x) exit'
    print(what_to_do)
    answer = input()
    while loop:
        if answer == 'e':
            deeplabcut.extract_frames(config_pathma, mode='automatic', algo='kmeans', userfeedback=False, cluster_resizewidth=10, cluster_step=1)
            print(what_to_do)
            answer = input()
        elif answer == 'l':
            deeplabcut.label_frames(config_pathma)
            print(what_to_do)
            answer = input()
        elif answer == 't':
            deeplabcut.create_multianimaltraining_dataset(config_pathma, net_type='resnet_50')
            print('max iterations')
            iterations = input()
            deeplabcut.train_network(config_pathma, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0)
            deeplabcut.evaluate_network(config_pathma, gputouse=0, plotting=True)
            pbounds = {
                        'pafthreshold': (0.05, 0.7),
                        'detectionthresholdsquare': (0, 0.9),
                        'minimalnumberofconnections': (1, 6),
                    }
            deeplabcut.evaluate_multianimal_crossvalidate(config_pathma, pbounds=pbounds, target='pck_test')
            print(what_to_do)
            answer = input()
        elif answer == 'a':
            scorername = deeplabcut.analyze_videos(config_pathma, [videos_dir], videotype='.mp4', gputouse=0, save_as_csv=False)
            #deeplabcut.create_video_with_all_detections(config_pathma, [videos_dir], DLCscorername=scorername)
            deeplabcut.convert_detections2tracklets(config_pathma, [videos_dir], videotype='.mp4', track_method='skeleton')
            videos = range(len(glob.glob(videos_dir+'\*_sk.pickle')))
            for video in videos:
                man, viz = deeplabcut.refine_tracklets(config_pathma, os.path.join(videos_dir, glob.glob(videos_dir+'\*_sk.pickle')[video]), os.path.join(file_dialog('vfile')))
            print(what_to_do)
            answer = input()
        elif answer == 'c':
            vids = range(len(glob.glob(videos_dir+'\*_sk.h5')))
            for vid in vids:
                ma_h5 = pd.read_hdf(glob.glob(videos_dir+'\*_sk.h5')[vid])
                sa_h5 = pd.read_hdf(glob.glob(str(Path(videos_dir).parent.parent)+'\other\*_'+str(iterations)+'.h5')[vid])
                outfile = os.path.join(videos_dir, os.path.basename(glob.glob(str(Path(videos_dir).parent.parent)+'\other\*'+str(iterations)+'.h5')[vid]))
                ma_h5.columns = sa_h5.columns
                ma_h5.to_hdf(outfile, key="df_with_missing", mode="w")
            deeplabcut.triangulate(config_path3d, videos_dir, videotype='.mp4', gputouse=0, filterpredictions=True)
            print(what_to_do)
            answer = input()
        elif answer == 'p':
            deeplabcut.create_labeled_video_3d(config_path3d, [videos_dir], videotype='.mp4', trailpoints=10, view=[0,270])
            print(what_to_do)
            answer = input()
        elif answer == 'r':
            deeplabcut.extract_outlier_frames(config_pathma, [videos_dir], videotype='.mp4', extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True, track_method='box')
            deeplabcut.refine_labels(config_pathma)
            print()
        elif answer == 'm':
            print('max iterations')
            iterations = input()
            deeplabcut.merge_datasets(config_pathma)
            deeplabcut.create_multianimaltraining_dataset(config_pathma, net_type='resnet_50')
            deeplabcut.train_network(config_pathma, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0)
            deeplabcut.evaluate_network(config_pathma, gputouse=0, plotting=True)
            deeplabcut.evaluate_multianimal_crossvalidate(config_pathma, plotting=True)
            print(what_to_do)
            answer = input()
        elif answer == 'v':
            new_videos_dir = os.path.join(file_dialog('directory'))
            deeplabcut.add_new_videos(config_pathma, [new_videos_dir], copy_videos=False)
            print(what_to_do)
            answer = input()
        elif answer == 'x':
            loop = False
        else:
            print('error')
            answer = input()


def print_config():
    print('videos_dir\t', videos_dir)
    print('config_path\t', config_path)
    print('config_pathma\t', config_pathma)
    print('config_path3d\t', config_path3d)


config_path, config_pathma, config_path3d, videos_dir = open_project()

def run_workflow():
    loop = True
    what_to_do = '(p) config paths\n(w) workflow\n(x) exit'
    print(what_to_do)
    console = input()
    while loop:
        if console == 'p':
            print_config()
            print(what_to_do)
            console = input()
        elif console == 'w':
            workflow(config_path, config_pathma, config_path3d, videos_dir)
            print(what_to_do)
            console = input()
        elif console == 'x':
            loop = False
        else:
            print('error')
            console = input()
