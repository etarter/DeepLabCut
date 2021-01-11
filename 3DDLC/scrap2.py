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

    X_l = auxiliaryfunctions_3d.triangulatePoints(
        P1, P2, points_cam1_undistort, points_cam2_undistort
    )

    # ToDo: speed up func. below by saving in numpy.array
    X_final.append(X_l)

#-------------------------------------------------------------------------------

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
auxiliaryfunctions_3d.SaveMetadata3d(
    str(output_filename + "_meta.pickle"), metadata
)

if save_as_csv:
    df_3d.to_csv(str(output_filename + ".csv"))

print("Triangulated data for video", video_list[i])
print("Results are saved under: ", destfolder)
# have to make the dest folder none so that it can be updated for a new pair of videos
if destfolder == str(Path(video).parents[0]):
    destfolder = None
