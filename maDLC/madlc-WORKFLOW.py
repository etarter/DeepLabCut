#workflow for multianimal dlc

videos_dir = r'path_to_videos'

deeplabcut.create_new_project('projectname', 'experimentername', [videos_dir], videotype='.mp4', copy_videos=False, multianimal=True)

config_path = r'path_to_config'

'''
CHANGE IN CONFIG FILE
individuals
multianimalbodyparts
skeleton (more than you would think for training)
numframes2pick
dotsize
batchsize
'''

deeplabcut.add_new_videos(config_path, [new_videos_dir], copy_videos=False)

deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False)

deeplabcut.label_frames(config_path)

deeplabcut.create_multianimaltraining_dataset(config_path, net_type='resnet_50')

deeplabcut.train_network(config_path, displayiters=10, maxiters=1000, allow_growth=True, gputouse=0)

deeplabcut.evaluate_network(config_path, gputouse=0, plotting=True)

deeplabcut.evaluate_multianimal_crossvalidate(config_path, plotting=True)

deeplabcut.analyze_videos(config_path, [r'C:\Users\etarter\Downloads\videos2-multianimal'], videotype='.mp4', gputouse=0)

#optional for direct creation of video from detections
scorername = deeplabcut.analyze_videos(config_path, [r'C:\Users\etarter\Downloads\videos2-multianimal'], videotype='.mp4', gputouse=0)
video_path = r'full_path_to_video'
deeplabcut.create_video_with_all_detections(config_path, [video_path], DLCscorername=scorername)

#specify full video path otherwise video from detection raises an error
#creates _box.pickle file from _full.pickle file from previous step
video_path = r'full_path_to_video'
deeplabcut.convert_detections2tracklets(config_path, [video_path], videotype='.mp4', track_method='box')

#refine tracklets in gui over and over again
#creates .h5 file from .pickle file
#DOES NOT WORK IN SPYDER KERNEL
pickle_or_h5_path = r'full_path_to_pickle_or_h5_file'
video_path = r'full_path_to_video'
deeplabcut.refine_tracklets(config_path, pickle_or_h5_path, video_path)

#or just skip refinement GUI and convert tracklets directly
#creates .h5 file from .pickle file
pickle_path = r'full_path_to_pickle_file'
deeplabcut.convert_raw_tracks_to_h5(config_path, pickle_path)

#filter data
#only working if you modify .pickle file to scorername without "_full" or "_box" at the end
deeplabcut.filterpredictions(config_path, [videos_dir], videotype='.mp4')

#plot trajectories
#creates directory called plot-poses
deeplabcut.plot_trajectories(config_path, [videos_dir], videotype='.mp4', track_method='box')

#create labeled video
#only working if you modify .pickle file to scorername without "_full" or "_box" at the end
deeplabcut.create_labeled_video(config_path, [videos_dir], videotype='.mp4', draw_skeleton=True)

#extract skeleton bone angles
#NOT WORKING YET
deeplabcut.analyzeskeleton(config_path, [videos_dir], videotype='.mp4', track_method='box')

#extract outlier frames
#only working if you modify .pickle file to scorername without "_full" or "_box" at the end
deeplabcut.extract_outlier_frames(config_path, [videos_dir], videotype='.mp4', extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True)

#refine labels
deeplabcut.refine_labels(config_path)

#merge datasets
deeplabcut.merge_datasets(config_path)
