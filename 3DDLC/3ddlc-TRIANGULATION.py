import cv2
import numpy as np
import pandas as pd
import tkinter
from tkinter import filedialog
import os
import glob
from pathlib import Path
import pickle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from tqdm import tqdm
matplotlib_axes_logger.setLevel("ERROR")


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
    elif type == 'directory':
        path = filedialog.askdirectory(parent=root, initialdir=os.getcwd(), title='select video directory')     #add automatic video directory assignement
    root.withdraw()
    return path


def write_pickle(filename, data):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)


# def calibrate(cbrow, cbcol, calibrate=False, alpha=0.4):
# return vars
# vars = calibrate(5, 8, False, 0.5)


# parameters
calibrate = True
alpha = 0.8
cbrow = 5
cbcol = 8
img_path = r'C:\Users\etarter\Downloads\calibration'
cam1 = 'cam1'
cam2 = 'cam2'
corner_path = r'C:\Users\etarter\Downloads\calibration\corners'
path_camera_matrix = r'C:\Users\etarter\Downloads\calibration\camera_matrix'
path_undistort = r'C:\Users\etarter\Downloads\calibration\undistortion'
cmap = 'jet'
markerSize = 5
alphaValue = 0.8
markerType = '*'
markerColor = 'r'
pcutoff = 0.4
# camera names
# print('camera 1 name:')
# cam1 = input()
# print('camera 2 name:')
# cam2 = input()
cam_names = [cam1, cam2]
plot = True
video_path = r'C:\Users\etarter\Downloads\videos'
videotype = '.mp4'
gputouse = 0
save_as_csv = False
filterpredictions = True
filtertype = 'median'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# chessboard corner object
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# calibration images
# img_path = os.path.join(file_dialog('directory'))
images = glob.glob(os.path.join(img_path, "*.jpg"))

# calibration objects
img_shape = {}
objpoints = {}  # 3d point in real world space
imgpoints = {}  # 2d points in image plane.
dist_pickle = {}
stereo_params = {}
for cam in cam_names:
    objpoints.setdefault(cam, [])
    imgpoints.setdefault(cam, [])
    dist_pickle.setdefault(cam, [])

# sort images
images.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
if len(images) == 0:
    raise Exception('No calibration images found')


# -----------------------------------------------------------------------------
# extract chessboard corners
for fname in images:
    for cam in cam_names:
        if cam in fname:
            filename = Path(fname).stem
            img = cv2.imread(fname)
            ext = Path(fname).suffix
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                img_shape[cam] = gray.shape[::-1]
                objpoints[cam].append(objp)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints[cam].append(corners)
                img = cv2.drawChessboardCorners(img, (cbcol, cbrow), corners, ret)
                cv2.imwrite(os.path.join(str(corner_path), filename + "_corner.jpg"), img)
            else:
                print("no corners found for %s" % Path(fname).name)
try:
    h, w = img.shape[:2]
except:
    raise Exception('camera names do not match')
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# calibrate cameras
if calibrate == True:
    # Calibrating each camera
    for cam in cam_names:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints[cam], imgpoints[cam], img_shape[cam], None, None)

        # Save the camera calibration result for later use (we won't use rvecs / tvecs)
        dist_pickle[cam] = {
            'mtx': mtx,
            'dist': dist,
            'objpoints': objpoints[cam],
            'imgpoints': imgpoints[cam],
        }
        pickle.dump(
            dist_pickle,
            open(os.path.join(path_camera_matrix, cam + '_intrinsic_params.pickle'), 'wb', ), )
        print('saving intrinsic camera calibration matrices for %s as a pickle file in %s'% (cam, os.path.join(path_camera_matrix)))

        # Compute mean re-projection errors for individual cameras
        mean_error = 0
        for i in range(len(objpoints[cam])):
            imgpoints_proj, _ = cv2.projectPoints(objpoints[cam][i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[cam][i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
            mean_error += error
        print('mean re-projection error for %s images: %.3f pixels '% (cam, mean_error / len(objpoints[cam])))

    # Compute stereo calibration for each pair of cameras
    camera_pair = [[cam_names[0], cam_names[1]]]
    for pair in camera_pair:
        print('computing stereo calibration for ' % pair)
        (
            retval,
            cameraMatrix1,
            distCoeffs1,
            cameraMatrix2,
            distCoeffs2,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            objpoints[pair[0]],
            imgpoints[pair[0]],
            imgpoints[pair[1]],
            dist_pickle[pair[0]]['mtx'],
            dist_pickle[pair[0]]['dist'],
            dist_pickle[pair[1]]['mtx'],
            dist_pickle[pair[1]]['dist'],
            (h, w),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )

        # Stereo Rectification
        rectify_scale = (
            alpha
        )
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1,
            distCoeffs1,
            cameraMatrix2,
            distCoeffs2,
            (h, w),
            R,
            T,
            alpha=rectify_scale,
        )

        stereo_params[pair[0] + '-' + pair[1]] = {
            'cameraMatrix1': cameraMatrix1,
            'cameraMatrix2': cameraMatrix2,
            'distCoeffs1': distCoeffs1,
            'distCoeffs2': distCoeffs2,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'roi1': roi1,
            'roi2': roi2,
            'Q': Q,
            'image_shape': [img_shape[pair[0]], img_shape[pair[1]]],
        }

    print('saving stereo parameters for each camera %s' % str(os.path.join(path_camera_matrix)))

    write_pickle(os.path.join(path_camera_matrix, 'stereo_params.pickle'), stereo_params)
    print('calibration done, now check undistortion')
else:
    print('corners extracted to %s, check corners and re-run setting calibrate=True' % str(corner_path))
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# undistort images
stereo_params = read_pickle(os.path.join(path_camera_matrix, "stereo_params.pickle"))

for pair in camera_pair:
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        stereo_params[pair[0] + "-" + pair[1]]["cameraMatrix1"],
        stereo_params[pair[0] + "-" + pair[1]]["distCoeffs1"],
        stereo_params[pair[0] + "-" + pair[1]]["R1"],
        stereo_params[pair[0] + "-" + pair[1]]["P1"],
        (stereo_params[pair[0] + "-" + pair[1]]["image_shape"][0]),
        cv2.CV_16SC2,
    )
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        stereo_params[pair[0] + "-" + pair[1]]["cameraMatrix2"],
        stereo_params[pair[0] + "-" + pair[1]]["distCoeffs2"],
        stereo_params[pair[0] + "-" + pair[1]]["R2"],
        stereo_params[pair[0] + "-" + pair[1]]["P2"],
        (stereo_params[pair[0] + "-" + pair[1]]["image_shape"][1]),
        cv2.CV_16SC2,
    )
cam1_undistort = []
cam2_undistort = []

for fname in images:
    if pair[0] in fname:
        filename = Path(fname).stem
        img1 = cv2.imread(fname)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        h, w = img1.shape[:2]
        _, corners1 = cv2.findChessboardCorners(gray1, (cbcol, cbrow), None)
        corners_origin1 = cv2.cornerSubPix(
            gray1, corners1, (11, 11), (-1, -1), criteria
        )

        # Remapping dataFrame_camera1_undistort
        im_remapped1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LANCZOS4)
        imgpoints_proj_undistort = cv2.undistortPoints(
            src=corners_origin1,
            cameraMatrix=stereo_params[pair[0] + "-" + pair[1]][
                "cameraMatrix1"
            ],
            distCoeffs=stereo_params[pair[0] + "-" + pair[1]]["distCoeffs1"],
            P=stereo_params[pair[0] + "-" + pair[1]]["P1"],
            R=stereo_params[pair[0] + "-" + pair[1]]["R1"],
        )
        cam1_undistort.append(imgpoints_proj_undistort)
        cv2.imwrite(os.path.join(str(path_undistort), filename + "_undistort.jpg"),im_remapped1,)
        imgpoints_proj_undistort = []

    elif pair[1] in fname:
        filename = Path(fname).stem
        img2 = cv2.imread(fname)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        h, w = img2.shape[:2]
        _, corners2 = cv2.findChessboardCorners(gray2, (cbcol, cbrow), None)
        corners_origin2 = cv2.cornerSubPix(
            gray2, corners2, (11, 11), (-1, -1), criteria
        )

        # Remapping
        im_remapped2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LANCZOS4)
        imgpoints_proj_undistort2 = cv2.undistortPoints(
            src=corners_origin2,
            cameraMatrix=stereo_params[pair[0] + "-" + pair[1]][
                "cameraMatrix2"
            ],
            distCoeffs=stereo_params[pair[0] + "-" + pair[1]]["distCoeffs2"],
            P=stereo_params[pair[0] + "-" + pair[1]]["P2"],
            R=stereo_params[pair[0] + "-" + pair[1]]["R2"],
        )
        cam2_undistort.append(imgpoints_proj_undistort2)
        cv2.imwrite(os.path.join(str(path_undistort), filename + "_undistort.jpg"),im_remapped2,)
        imgpoints_proj_undistort2 = []

cam1_undistort = np.array(cam1_undistort)
cam2_undistort = np.array(cam2_undistort)
print("undistortion complete in %s" % str(path_undistort))
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# plot undistortion
if plot == True:
    f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f1.suptitle(
        str("Original Image: Views from " + pair[0] + " and " + pair[1]),
        fontsize=25,
    )

    # Display images in RGB
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    norm = mcolors.Normalize(vmin=0.0, vmax=cam1_undistort.shape[1])
    plt.savefig(os.path.join(str(path_undistort), "Original_Image.png"))

    # Plot the undistorted corner points
    f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f2.suptitle(
        "Undistorted corner points on camera-1 and camera-2", fontsize=25
    )
    ax1.imshow(cv2.cvtColor(im_remapped1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(im_remapped2, cv2.COLOR_BGR2RGB))
    for i in range(0, cam1_undistort.shape[1]):
        ax1.scatter(
            [cam1_undistort[-1][i, 0, 0]],
            [cam1_undistort[-1][i, 0, 1]],
            marker=markerType,
            s=markerSize,
            color=markerColor,
            alpha=alphaValue,
        )
        ax2.scatter(
            [cam2_undistort[-1][i, 0, 0]],
            [cam2_undistort[-1][i, 0, 1]],
            marker=markerType,
            s=markerSize,
            color=markerColor,
            alpha=alphaValue,
        )
    plt.savefig(os.path.join(str(path_undistort), "undistorted_points.png"))
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# triangulate undistorted points
def triangulatePoints(P1, P2, x1, x2):
    X = cv2.triangulatePoints(P1[:3], P2[:3], x1, x2)
    return X / X[3]

def compute_triangulation_calibration_images(stereo_matrix, projectedPoints1, projectedPoints2, path_undistort, plot=True):
    triangulate = []
    P1 = stereo_matrix["P1"]
    P2 = stereo_matrix["P2"]
    colormap = plt.get_cmap(cmap)

    for i in range(projectedPoints1.shape[0]):
        X_l = triangulatePoints(P1, P2, projectedPoints1[i], projectedPoints2[i])
        triangulate.append(X_l)
    triangulate = np.asanyarray(triangulate)

    # Plotting
    if plot == True:
        col = colormap(np.linspace(0, 1, triangulate.shape[0]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i in range(triangulate.shape[0]):
            xs = triangulate[i, 0, :]
            ys = triangulate[i, 1, :]
            zs = triangulate[i, 2, :]
            ax.scatter(xs, ys, zs, c=col[i], marker=markerType, s=markerSize)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        plt.savefig(os.path.join(str(path_undistort), "checkerboard_3d.png"))
    return triangulate

triangulate = compute_triangulation_calibration_images(
                stereo_params[pair[0] + "-" + pair[1]],
                cam1_undistort,
                cam2_undistort,
                path_undistort,
                plot=plot,
                )
write_pickle(os.path.join(img_path, "triangulate.pickle"), triangulate)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# END OF CALIBRATION
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# get videos in path corresponding to the camera names
def get_camerawise_videos(path, cam_names, videotype):
    vid = []

    # Find videos only specific to the cam names
    videos = [
        glob.glob(os.path.join(path, str("*" + cam_names[i] + "*" + videotype)))
        for i in range(len(cam_names))
    ]
    videos = [y for x in videos for y in x]

    # Exclude the labeled video files
    if "." in videotype:
        file_to_exclude = str("labeled" + videotype)
    else:
        file_to_exclude = str("labeled." + videotype)
    videos = [v for v in videos if os.path.isfile(v) and not (file_to_exclude in v)]
    video_list = []
    cam = cam_names[0]  # camera1
    vid.append(
        [
            name
            for name in glob.glob(os.path.join(path, str("*" + cam + "*" + videotype)))
        ]
    )  # all videos with cam
    # print("here is what I found",vid)
    for k in range(len(vid[0])):
        if cam in str(Path(vid[0][k]).stem):
            ending = Path(vid[0][k]).suffix
            pref = str(Path(vid[0][k]).stem).split(cam)[0]
            suf = str(Path(vid[0][k]).stem).split(cam)[1]
            if pref == "":
                if suf == "":
                    print("Strange naming convention on your part. Respect.")
                else:
                    putativecam2name = os.path.join(path, cam_names[1] + suf + ending)
            else:
                if suf == "":
                    putativecam2name = os.path.join(path, pref + cam_names[1] + ending)
                else:
                    putativecam2name = os.path.join(
                        path, pref + cam_names[1] + suf + ending
                    )
            # print([os.path.join(path,pref+cam+suf+ending),putativecam2name])
            if os.path.isfile(putativecam2name):
                # found a pair!!!
                video_list.append(
                    [os.path.join(path, pref + cam + suf + ending), putativecam2name]
                )
    return video_list
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# import videos from path
flag = False  # assumes that video path is a list
if isinstance(video_path, str) == True:
    flag = True
    video_list = get_camerawise_videos(
        video_path, cam_names, videotype=videotype
    )
else:
    video_list = video_path

if video_list == []:
    print('no videos found in path ', video_path)
    print('make sure videotype and camera names are specified')
    print('i was looking for ', videotype,)
print('video pairs', video_list)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# create_empty dataframe function
def create_empty_df(dataframe, scorer, flag):
    # Creates an empty dataFrame of same shape as df_side_view
    # flag = 2d or 3d

    df = dataframe
    bodyparts = df.columns.get_level_values(1)
    _, idx = np.unique(bodyparts, return_index=True)
    bodyparts = list(bodyparts[np.sort(idx)])
    a = np.empty((df.shape[0], 3))
    a[:] = np.nan
    dataFrame = None
    for bodypart in bodyparts:
        if flag == "2d":
            pdindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y", "likelihood"]],
                names=["scorer", "bodyparts", "coords"],
            )
        elif flag == "3d":
            pdindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y", "z"]],
                names=["scorer", "bodyparts", "coords"],
            )
        frame = pd.DataFrame(a, columns=pdindex, index=range(0, df.shape[0]))
        dataFrame = pd.concat([frame, dataFrame], axis=1)
    return (dataFrame, scorer, bodyparts)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# undistort points function
path_corners = corner_path
def undistort_points(dataframe, camera_pair, destfolder):
    # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
    dataframe_cam1 = pd.read_hdf(dataframe[0])
    dataframe_cam2 = pd.read_hdf(dataframe[1])
    scorer_cam1 = dataframe_cam1.columns.get_level_values(0)[0]
    scorer_cam2 = dataframe_cam2.columns.get_level_values(0)[0]
    stereo_file = read_pickle(os.path.join(path_camera_matrix, "stereo_params.pickle"))
    path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
    stereo_file = read_pickle(path_stereo_file)
    mtx_l = stereo_file[camera_pair]["cameraMatrix1"]
    dist_l = stereo_file[camera_pair]["distCoeffs1"]

    mtx_r = stereo_file[camera_pair]["cameraMatrix2"]
    dist_r = stereo_file[camera_pair]["distCoeffs2"]

    R1 = stereo_file[camera_pair]["R1"]
    P1 = stereo_file[camera_pair]["P1"]

    R2 = stereo_file[camera_pair]["R2"]
    P2 = stereo_file[camera_pair]["P2"]

    # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
    (
        dataFrame_cam1_undistort,
        scorer_cam1,
        bodyparts,
    ) = create_empty_df(
        dataframe_cam1, scorer_cam1, flag="2d"
    )
    (
        dataFrame_cam2_undistort,
        scorer_cam2,
        bodyparts,
    ) = create_empty_df(
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

    # Save the undistorted files
    dataFrame_cam1_undistort.sort_index(inplace=True)
    dataFrame_cam2_undistort.sort_index(inplace=True)

    return (
        dataFrame_cam1_undistort,
        dataFrame_cam2_undistort,
        stereo_file[camera_pair],
        path_stereo_file,
    )
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# save metadata to pickle
def SaveMetadata3d(metadatafilename, metadata):
    with open(metadatafilename, "wb") as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# h5 file modifications
# change column multiindex with
# dataframe_new = dataframe_ma
# dataframe_new.columns = dataframe_sa.columns
# dataframe_new.to_hdf(str(output_filename + ".h5"), "df_with_missing", format="table", mode="w",)
dataname = [os.path.join(r'C:\Users\etarter\Downloads\videos\cam1-me-vidDLC_resnet50_hybridNov23shuffle1_10000_filtered.h5'),
            os.path.join(r'CC:\Users\etarter\Downloads\videos\cam2-me-vidDLC_resnet50_hybridNov23shuffle1_10000_filtered.h5')]
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# run triangulation with dataname
run_triangulate = True
scorer_3d = 'DLC_3D'
scorer_name = 'DLC_resnet50_hybridNov23shuffle1_10000_filtered'
output_filename = 'me-vid_DLC_3D'
if run_triangulate:
    # if len(dataname)>0:
    # undistort points for this pair
    print("Undistorting...")
    (
        dataFrame_camera1_undistort,
        dataFrame_camera2_undistort,
        stereomatrix,
        path_stereo_file,
    ) = undistort_points(
        dataname, str(cam_names[0] + "-" + cam_names[1]), None
    )
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
    df_3d, scorer_3d, bodyparts = create_empty_df(
        dataFrame_camera1_undistort, scorer_3d, flag="3d"
    )
    P1 = stereomatrix["P1"]
    P2 = stereomatrix["P2"]

    print("Computing the triangulation...")
    for bpindex, bp in enumerate(bodyparts):
        # Extract the indices of frames where the likelihood of a bodypart for both cameras are less than pvalue
        likelihoods = np.array(
            [
                dataFrame_camera1_undistort[scorer_cam1][bp][
                    "likelihood"
                ].values[:],
                dataFrame_camera2_undistort[scorer_cam2][bp][
                    "likelihood"
                ].values[:],
            ]
        )
        likelihoods = likelihoods.T

        # Extract frames where likelihood for both the views is less than the pcutoff
        low_likelihood_frames = np.any(likelihoods < pcutoff, axis=1)
        # low_likelihood_frames = np.all(likelihoods < pcutoff, axis=1)

        low_likelihood_frames = np.where(low_likelihood_frames == True)[0]
        points_cam1_undistort = np.array(
            [
                dataFrame_camera1_undistort[scorer_cam1][bp]["x"].values[:],
                dataFrame_camera1_undistort[scorer_cam1][bp]["y"].values[:],
            ]
        )
        points_cam1_undistort = points_cam1_undistort.T

        # For cam1 camera: Assign nans to x and y values of a bodypart where the likelihood for is less than pvalue
        points_cam1_undistort[low_likelihood_frames] = np.nan, np.nan
        points_cam1_undistort = np.expand_dims(points_cam1_undistort, axis=1)

        points_cam2_undistort = np.array(
            [
                dataFrame_camera2_undistort[scorer_cam2][bp]["x"].values[:],
                dataFrame_camera2_undistort[scorer_cam2][bp]["y"].values[:],
            ]
        )
        points_cam2_undistort = points_cam2_undistort.T

        # For cam2 camera: Assign nans to x and y values of a bodypart where the likelihood is less than pvalue
        points_cam2_undistort[low_likelihood_frames] = np.nan, np.nan
        points_cam2_undistort = np.expand_dims(points_cam2_undistort, axis=1)

        X_l = triangulatePoints(
            P1, P2, points_cam1_undistort, points_cam2_undistort
        )

        # ToDo: speed up func. below by saving in numpy.array
        X_final.append(X_l)
    triangulate.append(X_final)
    triangulate = np.asanyarray(triangulate)
    metadata = {}
    metadata["stereo_matrix"] = stereomatrix
    metadata["stereo_matrix_file"] = path_stereo_file
    metadata["scorer_name"] = {
        cam_names[0]: scorer_name[cam_names[0]],
        cam_names[1]: scorer_name[cam_names[1]],
    }

    # Create an empty dataframe to store x,y,z of 3d data
    for bpindex, bp in enumerate(bodyparts):
        df_3d.iloc[:][scorer_3d, bp, "x"] = triangulate[0, bpindex, 0, :]
        df_3d.iloc[:][scorer_3d, bp, "y"] = triangulate[0, bpindex, 1, :]
        df_3d.iloc[:][scorer_3d, bp, "z"] = triangulate[0, bpindex, 2, :]

    df_3d.to_hdf(
        str(output_filename + ".h5"),
        "df_with_missing",
        format="table",
        mode="w",
    )
    SaveMetadata3d(
        str(output_filename + "_meta.pickle"), metadata
    )

    if save_as_csv:
        df_3d.to_csv(str(output_filename + ".csv"))

    print("Triangulated data for video", video_list[i])
# -----------------------------------------------------------------------------
