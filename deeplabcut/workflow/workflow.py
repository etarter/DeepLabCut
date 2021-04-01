import deeplabcut
import tkinter
from tkinter import filedialog
import subprocess
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import cv2
import os
import numpy as np
from tqdm import tqdm
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d
from glob import glob

matplotlib_axes_logger.setLevel("ERROR")
plt.close('all')

def workflow():

    config_pathma, config_path3d, videos_dir, realworld_dir, video_type = open_project()

    loop = True
    menu_1_0 = ['',
                '(p) print paths',
                '(w) workflow',
                '(x) exit',
                '\n']
    console = input('\n'.join(menu_1_0))

    while loop:
        if console == 'p':
            print_paths(config_pathma, config_path3d, videos_dir, realworld_dir)
            console = input('\n'.join(menu_1_0))

        elif console == 'w':
            process(config_pathma, config_path3d, videos_dir, realworld_dir, video_type)
            console = input('\n'.join(menu_1_0))

        elif console == 'x':
            loop = False

        else:
            print('error\n')
            console = input('\n'.join(menu_1_0))

    return config_pathma, config_path3d, videos_dir, realworld_dir


def open_project():

    menu_0_0 = ['',
                '(n) new project',
                '(l) load project',
                '\n']
    console = input('\n'.join(menu_0_0))
    loop = True

    while loop:
        if console == 'n':
            project_name = input('project name: ')
            your_name = input('your name: ')
            loop_1 = True
            menu_0_1 = ['',
                        'videotype',
                        '(m) mp4',
                        '(a) avi',
                        '\n']
            video_type = input('\n'.join(menu_0_1))
            while loop_1:
                if video_type == 'm':
                    video_type = '.mp4'
                    loop_1 = False

                elif video_type == 'a':
                    video_type = '.avi'
                    loop_1 = False
                else:
                    print('error\n')
                    video_type = input('\n'.join(menu_0_1))
            videos_dir = os.path.join(file_dialog('dir', 'select video directory'))
            realworld_dir = os.path.join(file_dialog('dir', 'select realworld directory'))
            config_pathma = deeplabcut.create_new_project(project_name, your_name, [videos_dir], videotype=video_type, copy_videos=False, multianimal=True)
            config_path3d = deeplabcut.create_new_project_3d(project_name, your_name, num_cameras=2)
            print('edit config files')
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_pathma])
            subprocess.call([r'C:\Users\etarter\AppData\Local\atom\atom.exe', config_path3d])
            loop = False

        elif console == 'l':
            loop_1 = True
            menu_0_2 = ['',
                        '(n) load new',
                        '(l) load existing',
                        '\n']
            model = input('\n'.join(menu_0_2))

            while loop_1:
                if model == 'l':
                    config_pathma = r'C:\Users\etarter\Documents\Playground\Calibration Depth\dlc\calibration-depth-dlc-2021-03-10\config.yaml'
                    config_path3d = r'C:\Users\etarter\Documents\Playground\Calibration Depth\dlc\calibration-depth-dlc-2021-03-09-3d\config.yaml'
                    videos_dir = r'C:\Users\etarter\Documents\Playground\Calibration Depth\videos'
                    realworld_dir = r'C:\Users\etarter\Documents\Playground\Calibration Depth\realworld'
                    video_type = '.avi'
                    loop_1 = False

                elif model == 'n':
                    config_pathma = os.path.join(file_dialog('f', 'select multianimal config file'))
                    config_path3d = os.path.join(file_dialog('f', 'select 3d config file'))
                    videos_dir = os.path.join(file_dialog('dir', 'select videos dir'))
                    realworld_dir = os.path.join(file_dialog('dir', 'select realworld directory'))
                    files = glob(os.path.join(videos_dir, '*.avi'))
                    if len(files) == 0:
                        video_type = '.mp4'
                    else:
                        video_type = '.avi'
                    loop_1 = False

                else:
                    print('error\n')
                    model = input('\n'.join(menu_0_2))
            loop = False

        else:
            print('error\n')
            console = input('\n'.join(menu_0_0))

    return config_pathma, config_path3d, videos_dir, realworld_dir, video_type


def process(config_pathma, config_path3d, videos_dir, realworld_dir, video_type):

    menu_2_0 = ['',
                'main functions\n',
                '(ef) extract frames',
                '(lf) label frames',
                '(tn) create, train and evaluate network',
                '(cv) cross-validate tracking parameters',
                '(av) analyze videos',
                '(rt) refine tracklets',
                '(fp) filter predictions',
                '(tp) triangulate',
                '(la) extract limb angles',
                '\nextra functions\n',
                '(cc) calibrate cameras',
                '(rw) set realworld coordinates',
                '(tr) triangulate realworld coordinates',
                '(dt) convert detections to tracklets',
                '(lv) create labeled video',
                '(eo) extract and refine outlier frames',
                '(mn) retrain network',
                '(nv) add new videos',
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
            answer = input('\n'.join(menu_2_0))

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

        elif answer == 'fp':
            loop_1 = True
            menu_2_1 = ['',
                        'filtertype',
                        '(a) arima',
                        '(m) median',
                        '\n']
            filter_type = input('\n'.join(menu_2_1))
            while loop_1:
                if filter_type == 'a':
                    filter_type = 'arima'
                    loop_1 = False

                elif filter_type == 'm':
                    filter_type = 'median'
                    loop_1 = False
                else:
                    print('error\n')
                    filter_type = input('\n'.join(menu_2_1))
            deeplabcut.filterpredictions(config_pathma, [videos_dir], videotype=video_type, filtertype=filter_type, track_method='skeleton', save_as_csv=False)
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

        elif answer == 'la':
            print('not implemented yet')
            answer = input('\n'.join(menu_2_0))

        elif answer == 'cc':
            cbrows = input('chessboard rows: ')
            cbcols = input('chessboard columns: ')
            deeplabcut.calibrate_cameras(config_path3d, cbrow=cbrows, cbcol=cbcols, calibrate=False, alpha=0.1)
            print('check extracted corners')
            os.startfile(os.path.join(os.path.dirname(config_path3d), 'corners'))

            loop_2 = True
            menu_2_2 = ['',
                        'calibrate',
                        '(c) start calibration',
                        '(x) exit',
                        '\n']
            calib = input('\n'.join(menu_2_2))

            while loop_2:
                if calib == 'c':
                    deeplabcut.calibrate_cameras(config_path3d, cbrow=cbrows, cbcol=cbcols, calibrate=True, alpha=0.1)
                    print('check undistortion')
                    os.startfile(os.path.join(os.path.dirname(config_path3d), 'undistortion'))
                    loop_2 = False

                elif calib == 'x':
                    loop_2 = False

                else:
                    print('error\n')
                    calib = input('\n'.join(menu_2_2))

            answer = input('\n'.join(menu_2_0))

        elif answer == 'rw':
            realworld(realworld_dir)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'tr':
            triangulate_realworld(config_path3d, videos_dir, realworld_dir)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'dt':
            deeplabcut.convert_detections2tracklets(config_pathma, [videos_dir], videotype=video_type, track_method='skeleton')
            pickles = glob(videos_dir+'/*_sk.pickle')
            shape = range(len(pickles))

            for i in shape:
                deeplabcut.convert_raw_tracks_to_h5(config_pathma, pickles[i])

            answer = input('\n'.join(menu_2_0))

        elif answer == 'lv':
            #config3d file needs to have synthetic skeleton for this to work!!
            #example: instead of head and dock, individual1_head, individual2_dock etc. for each individual separately!
            deeplabcut.create_labeled_video_3d(config_path3d, [videos_dir], videotype=video_type)
            answer = input('\n'.join(menu_2_0))

        elif answer == 'eo':
            deeplabcut.extract_outlier_frames(config_pathma, [videos_dir], videotype=video_type, extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True, track_method='skeleton')
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
            answer = input('error\n')


def print_paths(config_pathma, config_path3d, videos_dir, realworld_dir):

    print('\nconfig_pathma\t', config_pathma, '\nconfig_path3d\t', config_path3d, '\nvideos_dir\t', videos_dir, '\nrealworld_dir\t', realworld_dir)


def file_dialog(type, text):

    root = tkinter.Tk()
    root.geometry('0x0+0+0')
    root.lift()
    root.focus_force()
    root.deiconify()
    root.update_idletasks()
    if type == 'f':
        path = filedialog.askopenfilename(parent=root, initialdir=os.getcwd(), title=text)
    elif type == 'dir':
        path = filedialog.askdirectory(parent=root, initialdir=os.path.dirname(os.getcwd()), title=text)
    root.withdraw()

    return path


def realworld(realworld_dir):

    class RealWorld():
        def __init__(self):
            self.coords = []
            self.counter = 0

        def onclick(self, event):
            x, y = event.xdata, event.ydata
            self.coords.append([self.counter, x, y])
            ax.plot([x], [y], 'bx')
            plt.annotate(str(self.counter), (x + 10, y))
            self.counter += 1
            plt.show()

    images = glob(realworld_dir + '\*.jpg')
    existing = glob(realworld_dir + '\*.csv')

    if len(existing) == 0:
        for image in images:
            realWorld = RealWorld()

            plt.close('all')

            img = cv2.imread(image)

            fig, ax = plt.subplots()

            plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            fig.canvas.mpl_connect('button_press_event', realWorld.onclick)
            plt.show()

            coords_df = pd.DataFrame(realWorld.coords, columns=['point', 'x', 'y'])
            coords_df.to_csv(os.path.join(realworld_dir, os.path.basename(image).split('.')[0]+'_realworld.csv'), index=False)

            fig, ax = plt.subplots()
            for i in range(coords_df.shape[0]):
                x = coords_df.x.iloc[i]
                y = coords_df.y.iloc[i]
                ax.plot([x], [y], 'bx')
                plt.annotate(str(i), (x + 10, y))

            plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            plt.savefig(os.path.join(realworld_dir, os.path.basename(image).split('.')[0]+'_realworld.jpg'))

    else:
        print('\nreal world coordinates already extracted!\n')


def triangulate_realworld(config_path3d, videos_dir, realworld_dir):

    existing = glob(realworld_dir + '\*_realworld3D.csv')

    if len(existing) == 0:
        realworld_coords = glob(realworld_dir + '\*_realworld.csv')
        frame_no = 500

        size = len(realworld_coords)

        for i in range(int(size/2)):
            if i == 0:
                realworld_cam1_df = pd.read_csv(realworld_coords[i]).sort_values(by=['point'])
                realworld_cam2_df = pd.read_csv(realworld_coords[int(size/2)+i]).sort_values(by=['point'])
            else:
                next_cam1 = pd.read_csv(realworld_coords[i]).sort_values(by=['point'])
                next_cam2 = pd.read_csv(realworld_coords[int(size/2)+i]).sort_values(by=['point'])
                realworld_cam1_df = realworld_cam1_df.append(next_cam1, ignore_index=True)
                realworld_cam2_df = realworld_cam2_df.append(next_cam2, ignore_index=True)

        realworld_cam1 = np.array(realworld_cam1_df.iloc[:,1:])
        realworld_cam2 = np.array(realworld_cam2_df.iloc[:,1:])

        cfg_3d = auxiliaryfunctions.read_config(config_path3d)
        img_path, path_corners, path_camera_matrix, path_undistort = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)

        cam_names = cfg_3d["camera_names"]
        camera_pair = str(cam_names[0] + "-" + cam_names[1])

        path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
        stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)

        mtx_l = stereo_file[camera_pair]["cameraMatrix1"]
        dist_l = stereo_file[camera_pair]["distCoeffs1"]

        mtx_r = stereo_file[camera_pair]["cameraMatrix2"]
        dist_r = stereo_file[camera_pair]["distCoeffs2"]

        R1 = stereo_file[camera_pair]["R1"]
        P1 = stereo_file[camera_pair]["P1"]

        R2 = stereo_file[camera_pair]["R2"]
        P2 = stereo_file[camera_pair]["P2"]

        realworld_cam1_undistort = cv2.undistortPoints(
            src=realworld_cam1.astype(np.float32),
            cameraMatrix=mtx_l,
            distCoeffs=dist_l,
            P=P1,
            R=R1,
        )

        realworld_cam2_undistort = cv2.undistortPoints(
            src=realworld_cam2.astype(np.float32),
            cameraMatrix=mtx_r,
            distCoeffs=dist_r,
            P=P2,
            R=R2,
        )

        filename_3d = glob(videos_dir + '\*3D.h5')[0]
        trackingfile_3d_df = pd.read_hdf(filename_3d).iloc[frame_no:frame_no+1]

        realworld_3d = auxiliaryfunctions_3d.triangulatePoints(P1[:3], P2[:3], realworld_cam1_undistort, realworld_cam2_undistort)[:-1, :].T
        realworld_3d_df = pd.DataFrame(realworld_3d, columns=['x', 'y', 'z'])
        realworld_3d_df.insert(0, realworld_cam1_df.columns[0], realworld_cam1_df.point)
        realworld_3d_df.to_csv(os.path.join(realworld_dir, os.path.basename(filename_3d).split('.')[0] + '_realworld3D.csv'))

        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = list(realworld_3d_df.x)
        y = list(realworld_3d_df.y)
        z = list(realworld_3d_df.z)

        points = list(realworld_3d_df.point)

        size = len(points)

        columns = trackingfile_3d_df.columns.to_frame(index=False)
        bps = columns.bodyparts.unique()

        for bp in bps:
            x.append(trackingfile_3d_df.DLC_3D[bp].x.iloc[0])
            y.append(trackingfile_3d_df.DLC_3D[bp].y.iloc[0])
            z.append(trackingfile_3d_df.DLC_3D[bp].z.iloc[0])

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        ax.scatter(x[:size], y[:size], z[:size], c='b', marker='s', label='realworld')
        for i in range(size):
            ax.text(x[i], y[i], z[i], str(points[i]), None)

        ax.scatter(x[size:], y[size:], z[size:], c='r', marker='o', label='dlc')
        for i in range(size, len(x)):
            ax.text(x[i], y[i], z[i], str(i-size), None)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend(loc='upper left');
        plt.show()

    else:
        print('\nreal world coordinates already triangulated!\n')
