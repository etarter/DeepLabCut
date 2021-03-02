import deeplabcut
import tkinter
from tkinter import filedialog
import os
from glob import glob
import subprocess
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def workflow():

    config_pathma, config_path3d, videos_dir = open_project()

    loop = True
    what_to_do = '(p) config paths\n(w) workflow\n(x) exit'
    print(what_to_do)
    console = input()

    while loop:
        if console == 'p':
            print_config(config_pathma, config_path3d, videos_dir)
            print(what_to_do)
            console = input()

        elif console == 'w':
            process(config_pathma, config_path3d, videos_dir)
            print(what_to_do)
            console = input()

        elif console == 'x':
            loop = False

        else:
            print('error')
            console = input()


def print_config(config_pathma, config_path3d, videos_dir):

    print('config_pathma\t', config_pathma)
    print('config_path3d\t', config_path3d)
    print('videos_dir\t', videos_dir)


def file_dialog(type, text):

    root = tkinter.Tk()
    root.geometry('0x0+0+0')
    root.lift()
    root.focus_force()
    root.deiconify()
    root.update_idletasks()
    if type == 'f':
        path = filedialog.askopenfilename(parent=root, initialdir= os.getcwd(), title=text)
    elif type == 'dir':
        path = filedialog.askdirectory(parent=root, initialdir=os.getcwd(), title=text)
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
            videos_dir = os.path.join(file_dialog('dir', 'select video directory'))
            config_pathma = create_new_project(project_name, your_name, [videos_dir], videotype='.mp4', copy_videos=False, multianimal=True)
            config_path3d = create_new_project_3d(project_name, your_name, num_cameras=2)
            print('edit config files')
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_pathma])
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_path3d])
            loop = False

        elif console == 'l':
            model = 'test'

            if model == 'test':
                config_pathma = r'C:\Users\etarter\Downloads\dlc\test-dlc-2021-01-04\config.yaml'
                config_path3d = r'C:\Users\etarter\Downloads\dlc\test1-dlc-2021-01-05-3d\config.yaml'
                videos_dir = r'C:\Users\etarter\Downloads\videos'

            elif model == 'ask':
                config_pathma = os.path.join(file_dialog('f', 'select multianimal config file'))
                config_path3d = os.path.join(file_dialog('f', 'select 3d config file'))
                videos_dir = os.path.join(file_dialog('dir', 'select videos dir'))
            loop = False

        else:
            print('try again')
            console = input()

    return config_pathma, config_path3d, videos_dir


def process(config_pathma, config_path3d, videos_dir):

    loop = True
    what_to_do = '(e) extract frames\n(l) label frames\n(t) create, train and evaluate network \n(a) analyze videos\n(rt) refine tracklets or (dt) convert detections to tracklets\n(f) filter predictions\n(c) triangulate\n(p) create labeled video\n(r) extract outliers and refine labels\n(m) merge and retrain network\n(v) add new videos\n(x) exit\n'
    answer = input(what_to_do)

    while loop:
        if answer == 'e':
            extract_frames(config_pathma, mode='automatic', algo='kmeans', userfeedback=False, cluster_resizewidth=10, cluster_step=1)
            answer = input(what_to_do)

        elif answer == 'l':
            label_frames(config_pathma)
            answer = input(what_to_do)

        elif answer == 't':
            create_multianimaltraining_dataset(config_pathma, net_type='resnet_50')
            iterations = input('max iterations: ')
            save_iterations = input('save iterations: ')
            train_network(config_pathma, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0, saveiters=save_iterations)
            evaluate_network(config_pathma, gputouse=0, plotting=True)
            pbounds = {
                        'pafthreshold': (0.05, 0.7),
                        'detectionthresholdsquare': (0, 0.9),
                        'minimalnumberofconnections': (1, 6),
                    }
            evaluate_multianimal_crossvalidate(config_pathma, pbounds=pbounds, target='rpck_test')
            answer = input(what_to_do)

        elif answer == 'a':
            analyze_videos(config_pathma, [videos_dir], videotype='.mp4', gputouse=0, save_as_csv=False)
            answer = input(what_to_do)

        elif answer == 'rt':
            convert_detections2tracklets(config_pathma, [videos_dir], videotype='.mp4', track_method='skeleton')
            pickles = glob(videos_dir+'/*_sk.pickle')
            pickles.sort()
            videos = glob(videos_dir+'/*.mp4')
            videos.sort()
            shape = range(len(pickles))

            for i in shape:
                man, viz = refine_tracklets(config_pathma, pickles[i], videos[i])

            answer = input(what_to_do)

        elif answer == 'dt':
            convert_detections2tracklets(config_pathma, [videos_dir], videotype='.mp4', track_method='skeleton')
            pickles = glob(videos_dir+'/*_sk.pickle')
            shape = range(len(pickles))

            for i in shape:
                convert_raw_tracks_to_h5(config_pathma, pickles[i])

            answer = input(what_to_do)

        elif answer == 'f':
            filterpredictions(config_pathma, [videos_dir], videotype='.mp4', filtertype='arima', track_method='skeleton', save_as_csv=False)
            answer = input(what_to_do)

        elif answer == 'c':
            try:
                vids = glob(videos_dir+'/*_filtered.h5')

                for vid in vids:
                    tracking_file = pd.read_hdf(vid)
                    outfile = '_'.join(vid.split('_')[:-2])
                    columns = tracking_file.columns.to_frame(index=False)
                    shape = range(columns.shape[0])

                    for row in shape:
                        columns.bodyparts.iloc[row] = columns.individuals.iloc[row]+'_'+columns.bodyparts.iloc[row]

                    columns_syn = pd.MultiIndex.from_frame(columns.drop(columns=['individuals']))
                    tracking_file.to_hdf(outfile+'_original.h5', key="df_with_missing", mode="w")
                    tracking_file.columns = columns_syn
                    tracking_file.to_hdf(outfile+'_synthetic.h5', key="df_with_missing", mode="w")
                    tracking_file.to_hdf(outfile+'.h5', key="df_with_missing", mode="w")

            except AttributeError:
                print('files already modified!')

            triangulate(config_path3d, videos_dir, videotype='.mp4', gputouse=0, filterpredictions=True)
            vids_3d = glob(videos_dir+'/*3D.h5')

            for vid_3d in vids_3d:
                tracking_file_3d = pd.read_hdf(vid_3d)
                outfile_3d = vid_3d.split('.')[:-1][0]
                columns_3d = tracking_file_3d.columns.to_frame(index=False)
                individuals = []
                shape_3d = range(columns_3d.shape[0])

                for row in shape_3d:
                    id = columns_3d.bodyparts.iloc[row].split('_')[:1][0]
                    bp = '_'.join(columns_3d.bodyparts.iloc[row].split('_')[1:])
                    columns_3d.bodyparts.iloc[row] = bp
                    individuals.append(id)

                columns_3d.insert(1, 'individuals', individuals)
                columns_3d_syn = pd.MultiIndex.from_frame(columns_3d)
                tracking_file_3d.to_hdf(outfile_3d+'_original.h5', key="df_with_missing", mode="w")
                tracking_file_3d.columns = columns_3d_syn
                tracking_file_3d.to_hdf(outfile_3d+'_synthetic.h5', key="df_with_missing", mode="w")

            answer = input(what_to_do)

        elif answer == 'p':
            create_labeled_video_3d(config_path3d, [videos_dir], videotype='.mp4', trailpoints=10, view=[0,270])
            answer = input(what_to_do)

        elif answer == 'r':
            extract_outlier_frames(config_pathma, [videos_dir], videotype='.mp4', extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True, track_method='box')
            refine_labels(config_pathma)
            answer = input(what_to_do)

        elif answer == 'm':
            iterations = input('max iterations: ')
            merge_datasets(config_pathma)
            create_multianimaltraining_dataset(config_pathma, net_type='resnet_50')
            train_network(config_pathma, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0)
            evaluate_network(config_pathma, gputouse=0, plotting=True)
            evaluate_multianimal_crossvalidate(config_pathma, plotting=True)
            answer = input(what_to_do)

        elif answer == 'v':
            new_videos_dir = os.path.join(file_dialog('f', 'select video file'))
            add_new_videos(config_pathma, [new_videos_dir], copy_videos=False)
            answer = input(what_to_do)

        elif answer == 'x':
            loop = False

        else:
            answer = input('error: ')
