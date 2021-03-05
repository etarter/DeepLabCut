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

    config_pathma, config_path3d, videos_dir, video_type = open_project()

    loop = True
    menu_1_0 = ['',
                '(p) config paths',
                '(w) workflow',
                '(x) exit',
                '\n']
    console = input('\n'.join(menu_1_0))

    while loop:
        if console == 'p':
            print_config(config_pathma, config_path3d, videos_dir)
            console = input('\n'.join(menu_1_0))

        elif console == 'w':
            process(config_pathma, config_path3d, videos_dir)
            console = input('\n'.join(menu_1_0))

        elif console == 'x':
            loop = False

        else:
            print('error\n')
            console = input('\n'.join(menu_1_0))

    return config_pathma, config_path3d, videos_dir


def open_project():

    menu_0_0 = ['',
                '(n) new project', # todo add automatic 3d config file skeleton creation
                '(l) load project',
                '\n']
    console = input('\n'.join(menu_0_0))
    loop = True

    while loop:
        if console == 'n':
            project_name = input('project name: ')
            your_name = input('your name: ')
            video_type = input('video type: ')
            videos_dir = os.path.join(file_dialog('dir', 'select video directory'))
            config_pathma = deeplabcut.create_new_project(project_name, your_name, [videos_dir], videotype=video_type, copy_videos=False, multianimal=True)
            config_path3d = deeplabcut.create_new_project_3d(project_name, your_name, num_cameras=2)
            print('edit config files')
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_pathma])
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_path3d])
            loop = False

        elif console == 'l':
            loop = True
            menu_0_1 = ['',
                        '(n) load new',
                        '(l) load existing',
                        '\n']
            model = input('\n'.join(menu_0_1))

            while loop:
                if model == 'l':
                    config_pathma = r'C:\Users\etarter\Downloads\dlc\test-dlc-2021-01-04\config.yaml'
                    config_path3d = r'C:\Users\etarter\Downloads\dlc\test1-dlc-2021-01-05-3d\config.yaml'
                    videos_dir = r'C:\Users\etarter\Downloads\videos'
                    loop = False

                elif model == 'n':
                    config_pathma = os.path.join(file_dialog('f', 'select multianimal config file'))
                    config_path3d = os.path.join(file_dialog('f', 'select 3d config file'))
                    videos_dir = os.path.join(file_dialog('dir', 'select videos dir'))
                    loop = False

                else:
                    print('error\n')
                    model = input('\n'.join(menu_0_1))

        else:
            print('error\n')
            console = input('\n'.join(menu_0_0))

    return config_pathma, config_path3d, videos_dir, video_type


def process(config_pathma, config_path3d, videos_dir, video_type):

    menu_2_0 = ['',
                'main functions\n',
                '(ef) extract frames',
                '(lf) label frames',
                '(tn) create, train and evaluate network',
                '(cv) cross-validate tracking parameters', # todo: add more parameters and bounds
                '(av) analyze videos', # todo: select track method
                '(rt) refine tracklets',
                '(fp) filter predictions', # todo: select filter type
                '(tp) triangulate',
                '\nextra functions\n',
                '(cc) calibrate cameras', # todo: automatic directory opening to check for extracted corners
                '(dt) convert detections to tracklets',
                '(lv) create labeled video', # todo: select 2d or 3d labeled video
                '(eo) extract and refine outlier frames',
                '(mn) retrain network',
                '(nv) add new videos', # todo: test this for video or directory only?
                '\n(x) exit',
                '\n']
    answer = input('\n'.join(menu_2_0))
    loop = True

    while loop:
        if answer == 'ef':
            deeplabcut.extract_frames(config_pathma, mode='automatic', algo='kmeans', userfeedback=False, cluster_resizewidth=10, cluster_step=1)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'lf':
            deeplabcut.label_frames(config_pathma)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'tn':
            deeplabcut.create_multianimaltraining_dataset(config_pathma, net_type='resnet_50')
            iterations = input('max iterations: ')
            save_iterations = input('save iterations: ')
            deeplabcut.train_network(config_pathma, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0, saveiters=save_iterations)
            deeplabcut.evaluate_network(config_pathma, gputouse=0, plotting=True)

        elif answer == 'cv':
            pbounds = {
                        'pafthreshold': (0.05, 0.7),
                        'detectionthresholdsquare': (0, 0.9),
                        'minimalnumberofconnections': (1, 6),
                    }
            deeplabcut.evaluate_multianimal_crossvalidate(config_pathma, pbounds=pbounds, target='rpck_test')
            answer = input('\n'.join(menu_2_0))

        elif answer == 'av':
            deeplabcut.analyze_videos(config_pathma, [videos_dir], videotype=video_type, gputouse=0, save_as_csv=False)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'rt':
            deeplabcut.convert_detections2tracklets(config_pathma, [videos_dir], videotype=video_type, track_method='skeleton')
            pickles = glob(videos_dir+'/*_sk.pickle')
            pickles.sort()
            videos = glob(videos_dir+'/*.mp4')
            videos.sort()
            shape = range(len(pickles))

            for i in shape:
                man, viz = deeplabcut.refine_tracklets(config_pathma, pickles[i], videos[i])

            answer = input('\n'.join(menu_2_0))

        elif answer == 'dt':
            deeplabcut.convert_detections2tracklets(config_pathma, [videos_dir], videotype=video_type, track_method='skeleton')
            pickles = glob(videos_dir+'/*_sk.pickle')
            shape = range(len(pickles))

            for i in shape:
                deeplabcut.convert_raw_tracks_to_h5(config_pathma, pickles[i])

            answer = input('\n'.join(menu_2_0))

        elif answer == 'fp':
            deeplabcut.filterpredictions(config_pathma, [videos_dir], videotype=video_type, filtertype='arima', track_method='skeleton', save_as_csv=False)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'tp':
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

            deeplabcut.triangulate(config_path3d, videos_dir, videotype=video_type, gputouse=0, filterpredictions=True)
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

            answer = input('\n'.join(menu_2_0))

        elif answer == 'lv':
            deeplabcut.create_labeled_video_3d(config_path3d, [videos_dir], videotype=video_type, trailpoints=10, view=[0,270])
            answer = input('\n'.join(menu_2_0))

        elif answer == 'eo':
            deeplabcut.extract_outlier_frames(config_pathma, [videos_dir], videotype=video_type, extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True, track_method='box')
            deeplabcut.refine_labels(config_pathma)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'mn':
            deeplabcut.merge_datasets(config_pathma)
            answer = 'tn'

        elif answer == 'nv':
            new_videos_dir = os.path.join(file_dialog('dir', 'select video directory'))
            new_videos_list = glob(new_videos_dir+'/*.mp4')
            deeplabcut.add_new_videos(config_pathma, new_videos_list, copy_videos=False)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'x':
            loop = False

        else:
            answer = input('error: ')


def print_config(config_pathma, config_path3d, videos_dir):

    print('\nconfig_pathma\t', config_pathma, '\nconfig_path3d\t', config_path3d, '\nvideos_dir\t', videos_dir)


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
