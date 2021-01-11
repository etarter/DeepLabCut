def undistort_points(config, dataframe, camera_pair, destfolder):
    cfg_3d = auxiliaryfunctions.read_config(config)
    (
    img_path,
    path_corners,
    path_camera_matrix,
    path_undistort,
    ) = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)
    """
    path_undistort = destfolder
    filename_cam1 = Path(dataframe[0]).stem
    filename_cam2 = Path(dataframe[1]).stem

    #currently no interm. saving of this due to high speed.
    # check if the undistorted files are already present
    if os.path.exists(os.path.join(path_undistort,filename_cam1 + '_undistort.h5')) and os.path.exists(os.path.join(path_undistort,filename_cam2 + '_undistort.h5')):
    print("The undistorted files are already present at %s" % os.path.join(path_undistort,filename_cam1))
    dataFrame_cam1_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam1 + '_undistort.h5'))
    dataFrame_cam2_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam2 + '_undistort.h5'))
    else:
    """
    if True:
        # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
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

        # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
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

        # Save the undistorted files
        dataFrame_cam1_undistort.sort_index(inplace=True)
        dataFrame_cam2_undistort.sort_index(inplace=True)

    return (
        dataFrame_cam1_undistort,
        dataFrame_cam2_undistort,
        stereo_file[camera_pair],
        path_stereo_file,
    )
