# Single Animal DLC Workflow

## Description

The code will guide the user through the correct DLC workflow including fallback loops to minimize errors. The user can create or load a project by specifying the config file path in a popup dialog. From them the user will be prompted what methods should be executed: methods are speficic combination of DLC functions that should be executed in that precise order such that the project does not become corrupt. To exit the loop, restart kernel or press <kbd>Ctrl</kbd>+<kbd>C</kbd>.

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
config_path = deeplabcut.create_new_project('projectname', 'yourname', [videos_dir], copy_videos=True, videotype='.mp4', multianimal=False)
```
The code automatically modifies following config file parameters: `bodyparts`, `skeleton`, `dotsize` and `batchsize`. Now you can reload the project by specifying the config file path. It is also suggested you specify the video directory path.

New videos can always be added with:

```python
deeplabcut.add_new_videos(config_path, [new_videos_dir], copy_videos=False)
```

### Step 2

Extract frames for labelling:
```python
deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, cluster_resizewidth=10, cluster_step=1)
```
Relevant arguments:\
`userfeedback` automatically extracts the number of frames specified in the config file.
`cluster_resizewidth` automatically downscales image to specified width with fixed aspect ratio.
`cluster_step` is how many frames to skip during extraction, change this for long videos.
`cluster_color` whether frames should be analysed in color or grayscale.

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

Create training dataset:
```python
deeplabcut.create_training_dataset(config_path, net_type='resnet_50', augmenter_type='imgaug')
```
Relevant arguments:\
`augmenter_type` image augmentation with 'imgaug', 'tensorpack' or 'deterministic'

### Step 5

Start training:
```python
deeplabcut.train_network(config_path, displayiters=100, maxiters=10000, allow_growth=True, gputouse=0)
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

Extract scoremaps for all bodyparts:
```python
deeplabcut.extract_save_all_maps(config_path, gputouse=0)
```

### Step 6

Analyze videos:
```python
deeplabcut.analyze_videos(config_path,[videos_dir], videotype='.mp4', gputouse=0, batchsize=None, save_as_csv=False)
```
Relevant arguments:\
`save_as_csv` whether to save predictions as csv.\
`batchsize` specify batch size for improved inference.

### Step 6.1 (optional)

Create a labeled video:
```python
deeplabcut.create_labeled_video(config_path, [videos_dir], videotype='.mp4', draw_skeleton=True)
```
Relevant arguments:\
`draw_skeleton` also connects bodyparts according to defined skeleton in config file.

### Step 7

Refine network by extracting outlier frames:
```python
deeplabcut.extract_outlier_frames(config_path, [videos_dir], videotype='.mp4', extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True)
```
Relevant arguments:\
`automatic` extract number of frames specified in config file without user feedback.\
`cluster_resizewidth` automatically downscales image to specified width with fixed aspect ratio.\
`cluster_step` is how many frames to skip during extraction, change this for long videos.\
`cluster_color` whether frames should be analyzed in color or grayscale.

### Step 8

Refine networks by refining labels on extracted outlier frames:
```python
deeplabcut.refine_labels(config_path)
```

### Step 9

Merge, create and retrain network after refinement:
```python
deeplabcut.merge_datasets(config_path)
deeplabcut.create_training_dataset(config_path, net_type='resnet_50', augmenter_type='imgaug')
deeplabcut.train_network(config_path, displayiters=100, maxiters=10000, allow_growth=True, gputouse=0)
```
