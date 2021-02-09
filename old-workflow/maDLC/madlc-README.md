# Multianimal DLC Workflow

## Description

The code will guide the user through the correct DLC workflow including fallback loops to minimize errors. The user can create or load a project by specifying the config file path in a popup dialog. From them the user will be prompted what methods should be executed: methods are speficic combination of DLC functions that should be executed in that precise order such that the project does not become corrupt. To exit the loop, restart kernel or press <kbd>Ctrl</kbd>+<kbd>C</kbd>.

## Issues

Currently the code stops after executing ``refine_tracklets()`` due to some unknown issue. When executing line by line it works, but when executing it the code, it stops working after that step. This will hopefully get fixed soon.

## Usage

Run everything locally as long server links lead to errors and a directory junction does not work on mounted drives.

Activate GPU environment:
```bash
conda activate DLC-GPU
```
Or make sure to specify the environment interpreter in the IDE you are using i.e. in spyder, set interpreter path as `C:\Users\username\Anaconda3\envs\DLC-GPU\python.exe`

Get GPU number:
```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

## Workflow

### Step 1

Create a Project:
```python
import deeplabcut

videos_dir = r'path_to_videos'
config_path = deeplabcut.create_new_project(project_name, your_name, [videos_dir], videotype='.mp4', copy_videos=False, multianimal=True)
```
The code automatically modifies following config file parameters: `individuals`, `multianimalbodyparts`, `skeleton`, `dotsize`, `batchsize` and `numframes2pick`. Now you can reload the project by specifying the config file path. It is also suggested you specify the video directory path.

New videos can always be added with:

```python
deeplabcut.add_new_videos(config_path, [new_videos_dir], copy_videos=False)
```

### Step 2

Extract frames for labelling:
```python
deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False)
```
Relevant arguments:\
`userfeedback` automatically extracts the number of frames specified in the config file.

### Step 3

Label frames:
```python
deeplabcut.label_frames(config_path)
```
This will open the labelling GUI. Click "Load Frames" and specify the directory of the video for which you extracted frames.\
Hotkeys:\
<kbd>RMB</kbd> annotate\
<kbd>MMB</kbd> delete\
<kbd>LMB</kbd> drag

### Step 3.1 (optional)

Check labels before training:
```python
deeplabcut.check_labels(config_path)
```
This will create images with annotated labels. Check those in the `labeled-data` directory.

### Step 4

Create multianimal training dataset:
```python
deeplabcut.create_multianimaltraining_dataset(config_path, net_type='resnet_50')
```

### Step 5

Start training:
```python
deeplabcut.train_network(config_path, displayiters=10, maxiters=10000, allow_growth=True, gputouse=0)
```
Relevant arguments:\
`maxiters` how many iterations to train the network.\
`batchsize` can be changed in the `.../dlc-model/.../pose_cfg.yaml` file.\
`allow_growth` trick to not run out of vram.\
`gputouse` number of gpu from `device_lib`.

### Step 5.1 (optional)

Evaluate network
```python
deeplabcut.evaluate_network(config_path, gputouse=0, plotting=True)
```
Relevant arguments:
plotting plots predictions on test and train images

### Step 5.2 (optional)

Extract scoremaps for all bodyparts and individuals:
```python
deeplabcut.extract_save_all_maps(config_path, gputouse=0)
```

### Step 6

It is important for multianimal projects to crossvalidate parameters to achieve good detection performance.
```python
deeplabcut.evaluate_multianimal_crossvalidate(config_path, plotting=True)
```

### Step 7

Analyze videos and store scorename for optional direct labeled video creation from detections:
```python
scorername = deeplabcut.analyze_videos(config_path, [videos_dir], videotype='.mp4', gputouse=0, batchsize=16, save_as_csv=False)
```
Relevant arguments:\
`save_as_csv` whether to save predictions as csv.\
`batchsize` specify batch size for improved inference.

The difference for multianimal DLC is that detections need to be converted to tracklets. Tracking and bodypart detection are different steps and should be refined until tracking is satisfactory.

### Step 7.1 (optional)

Create labeled video directly from detections:

```python
deeplabcut.create_video_with_all_detections(config_path, [video_path], DLCscorername=scorername)
```

### Step 8

Create tracklets from detections:

```python
deeplabcut.convert_detections2tracklets(config_path, [videos_dir], videotype='.mp4', track_method='box')
```

### Step 9

Refine tracklets much like refining labels, but for increasing tracking accuracy:

```python
man, viz = deeplabcut.refine_tracklets(config_path, os.path.join(videos_dir, glob.glob(videos_dir+'\*_bx.pickle')[0]), os.path.join(file_dialog('vfile')))
```

The `man` and `viz` variables are just to omit command line output.

Note: this is where the code stops executing.

### Step 9.1 (optional)

Create a labeled video:
```python
deeplabcut.create_labeled_video(config_path, [videos_dir], videotype='.mp4', draw_skeleton=True, track_method='box')
```
Relevant arguments:\
`draw_skeleton` also connects bodyparts according to defined skeleton in config file.\
`track_method` needs to be specified for the correct tracking file to be assigned.

### Step 10

Refine network by extracting outlier frames:
```python
deeplabcut.extract_outlier_frames(config_path, [videos_dir], videotype='.mp4', extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True, track_method='box')
```
Relevant arguments:\
`automatic` extract number of frames specified in config file without user feedback.\
`track_method` needs to be specified for the correct tracking file to be assigned.

### Step 11

Refine networks by refining labels on extracted outlier frames:
```python
deeplabcut.refine_labels(config_path)
```

### Step 11.1 (optional)

Merge, create and retrain network after refinement:
```python
deeplabcut.merge_datasets(config_path)
deeplabcut.create_multianimaltraining_dataset(config_path, net_type='resnet_50')
deeplabcut.train_network(config_path, displayiters=10, maxiters=10000, allow_growth=True, gputouse=0)
```
