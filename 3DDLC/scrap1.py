from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path

#-------------------------------------------------------------------------------
#undistort_points() INPUT

dataname = [r'C:\Users\etarter\Downloads\videos\cam1-me-vidDLC_resnet50_testJan4shuffle1_10000_sk.h5', r'C:\Users\etarter\Downloads\videos\cam2-me-vidDLC_resnet50_testJan4shuffle1_10000_sk.h5']
config = config_path3d
dataframe = dataname
camera_pair = 'cam1-cam2'
destfolder = None
save_as_csv=False

#-------------------------------------------------------------------------------
#get scorername and stuff

scorer_name = {}
cfg_3d = auxiliaryfunctions.read_config(config)
video_path = videos_dir
videotype = '.mp4'
cam_names = cfg_3d["camera_names"]
pcutoff = cfg_3d["pcutoff"]
scorer_3d = cfg_3d["scorername_3d"]
video_list = auxiliaryfunctions_3d.get_camerawise_videos(video_path, cam_names, videotype=videotype)

snapshots = {}
for cam in cam_names:
    snapshots[cam] = cfg_3d[str("config_file_" + cam)]

flag = False  # assumes that video path is a list
if isinstance(video_path, str) == True:
    flag = True
    video_list = auxiliaryfunctions_3d.get_camerawise_videos(
        video_path, cam_names, videotype=videotype
    )
else:
    video_list = video_path

if video_list == []:
    print("No videos found in the specified video path.", video_path)
    print(
        "Please make sure that the video names are specified with correct camera names as entered in the config file or"
    )
    print(
        "perhaps the videotype is distinct from the videos in the path, I was looking for:",
        videotype,
    )

for i in range(len(video_list)):
    dataname = []
    for j in range(len(video_list[i])):
        if cam_names[j] in video_list[i][j]:
            config_2d = snapshots[cam_names[j]]
            cfg = auxiliaryfunctions.read_config(config_2d)
            shuffle = cfg_3d[str("shuffle_" + cam_names[j])]
            trainingsetindex = cfg_3d[str("trainingsetindex_" + cam_names[j])]
            trainFraction = cfg["TrainingFraction"][trainingsetindex]
            if flag == True:
                video = os.path.join(video_path, video_list[i][j])
            else:
                video_path = str(Path(video_list[i][j]).parents[0])
                video = os.path.join(video_path, video_list[i][j])

            if destfolder is None:
                destfolder = str(Path(video).parents[0])

            vname = Path(video).stem
            prefix = str(vname).split(cam_names[j])[0]
            suffix = str(vname).split(cam_names[j])[-1]
            if prefix == "":
                pass
            elif prefix[-1] == "_" or prefix[-1] == "-":
                prefix = prefix[:-1]

            if suffix == "":
                pass
            elif suffix[0] == "_" or suffix[0] == "-":
                suffix = suffix[1:]

            if prefix == "":
                output_file = os.path.join(destfolder, suffix)
            else:
                if suffix == "":
                    output_file = os.path.join(destfolder, prefix)
                else:
                    output_file = os.path.join(destfolder, prefix + "_" + suffix)
            output_filename = os.path.join(
                output_file + "_" + scorer_3d
            )
            DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
                cfg, shuffle, trainFraction, trainingsiterations="unknown"
            )
            scorer_name[cam_names[j]] = DLCscorer

#-------------------------------------------------------------------------------

(
img_path,
path_corners,
path_camera_matrix,
path_undistort,
) = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)



dataframe_cam1 = pd.read_hdf(dataframe[0])
dataframe_cam2 = pd.read_hdf(dataframe[1])

sa_cam1 = pd.read_hdf(r'C:\Users\etarter\Downloads\videos\test1\cam1-me-vidDLC_resnet50_test1Jan4shuffle1_10000.h5')
sa_cam2 = pd.read_hdf(r'C:\Users\etarter\Downloads\videos\test1\cam2-me-vidDLC_resnet50_test1Jan4shuffle1_10000.h5')

dataframe_cam1.columns = sa_cam1.columns
dataframe_cam2.columns = sa_cam2.columns

#dataframe_cam1 = sa_cam1
#dataframe_cam2 = sa_cam2

scorer_cam1 = dataframe_cam1.columns.get_level_values(0)[0]
scorer_cam2 = dataframe_cam2.columns.get_level_values(0)[0]
stereo_file = auxiliaryfunctions.read_pickle(
    os.path.join(path_camera_matrix, "stereo_params.pickle")
)
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

(
    dataFrame_cam1_undistort,
    scorer_cam1,
    bodyparts,
) = auxiliaryfunctions_3d.create_empty_df(
    dataframe_cam1, scorer_cam1, flag="2d"
)
(
    dataFrame_cam2_undistort,
    scorer_cam2,
    bodyparts,
) = auxiliaryfunctions_3d.create_empty_df(
    dataframe_cam2, scorer_cam2, flag="2d"
)

for bpindex, bp in tqdm(enumerate(bodyparts)):
    # Undistorting the points from cam1 camera
    points_cam1 = np.array(
        [
            dataframe_cam1[scorer_cam1][bp]["x"].values[:],
            dataframe_cam1[scorer_cam1][bp]["y"].values[:],
        ]
    )
    points_cam1 = points_cam1.T
    points_cam1 = np.expand_dims(points_cam1, axis=1)


    points_cam1 = np.float64(points_cam1)


    points_cam1_remapped = cv2.undistortPoints(
        src=points_cam1, cameraMatrix=mtx_l, distCoeffs=dist_l, P=P1, R=R1
    )

    dataFrame_cam1_undistort.iloc[:][
        scorer_cam1, bp, "x"
    ] = points_cam1_remapped[:, 0, 0]
    dataFrame_cam1_undistort.iloc[:][
        scorer_cam1, bp, "y"
    ] = points_cam1_remapped[:, 0, 1]
    dataFrame_cam1_undistort.iloc[:][
        scorer_cam1, bp, "likelihood"
    ] = dataframe_cam1[scorer_cam1][bp]["likelihood"].values[:]

    # Undistorting the points from cam2 camera
    points_cam2 = np.array(
        [
            dataframe_cam2[scorer_cam2][bp]["x"].values[:],
            dataframe_cam2[scorer_cam2][bp]["y"].values[:],
        ]
    )
    points_cam2 = points_cam2.T
    points_cam2 = np.expand_dims(points_cam2, axis=1)


    points_cam2 = np.float64(points_cam2)


    points_cam2_remapped = cv2.undistortPoints(
        src=points_cam2, cameraMatrix=mtx_r, distCoeffs=dist_r, P=P2, R=R2
    )

    dataFrame_cam2_undistort.iloc[:][
        scorer_cam2, bp, "x"
    ] = points_cam2_remapped[:, 0, 0]
    dataFrame_cam2_undistort.iloc[:][
        scorer_cam2, bp, "y"
    ] = points_cam2_remapped[:, 0, 1]
    dataFrame_cam2_undistort.iloc[:][
        scorer_cam2, bp, "likelihood"
    ] = dataframe_cam2[scorer_cam2][bp]["likelihood"].values[:]

dataFrame_cam1_undistort.sort_index(inplace=True)
dataFrame_cam2_undistort.sort_index(inplace=True)

#-------------------------------------------------------------------------------
#next code INPUT = undistort_points() OUTPUT

dataFrame_camera1_undistort = dataFrame_cam1_undistort
dataFrame_camera2_undistort = dataFrame_cam2_undistort
stereomatrix = stereo_file[camera_pair]
path_stereo_file = path_stereo_file

#-------------------------------------------------------------------------------

if len(dataFrame_camera1_undistort) != len(dataFrame_camera2_undistort):
    import warnings

    warnings.warn(
        "The number of frames do not match in the two videos. Please make sure that your videos have same number of frames and then retry! Excluding the extra frames from the longer video."
    )
    if len(dataFrame_camera1_undistort) > len(dataFrame_camera2_undistort):
        dataFrame_camera1_undistort = dataFrame_camera1_undistort[
            : len(dataFrame_camera2_undistort)
        ]
    if len(dataFrame_camera2_undistort) > len(dataFrame_camera1_undistort):
        dataFrame_camera2_undistort = dataFrame_camera2_undistort[
            : len(dataFrame_camera1_undistort)
        ]

X_final = []
triangulate = []
scorer_cam1 = dataFrame_camera1_undistort.columns.get_level_values(0)[0]
scorer_cam2 = dataFrame_camera2_undistort.columns.get_level_values(0)[0]
df_3d, scorer_3d, bodyparts = auxiliaryfunctions_3d.create_empty_df(
    dataFrame_camera1_undistort, scorer_3d, flag="3d"
)
P1 = stereomatrix["P1"]
P2 = stereomatrix["P2"]
