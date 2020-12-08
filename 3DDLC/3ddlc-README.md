### calibration image extraction
```bash
ffmpeg -i videoname.mp4 -vframes num_frames camname#-%03d.jpg
```

### important for creating a 3d labeled video
```python
from mpl_toolkits.mplot3d import Axes3D
```

### create 3d project
```python
config_path3d = deeplabcut.create_new_project_3d('projectname', 'yourname', num_cameras=2)
````
### edit config file
Change:
`skeleton`, `dotsize`, `camera_names`, `config_file_...`, `shuffle_...`, `trainingsetindex_...`

### link dlc config file in 3d dlc config file
`config_file_camname# = config_path`

### extract corners from calibration images
IMPORTANT: `cbrow` and `cbcol` correspond to the inner corners of the grid, and must be in the orientation you are holding the calibration pattern!
```python
deeplabcut.calibrate_cameras(config_path3d, cbrow=5, cbcol=8, calibrate=False, alpha=1)
```

### actually calibrate cameras
```python
deeplabcut.calibrate_cameras(config_path3d, cbrow=5, cbcol=8, calibrate=True, alpha=1)
```

### check undistortion
Check if camera calibration is satisfactory, the images should not be too warped.
```python
deeplabcut.check_undistortion(config_path3d, cbrow=5, cbcol=8)
```

### triangulate videos
This analyzes the video directory with the 2d config file specified in the 3d config file.
```python
video_dir = r'video_folder'
deeplabcut.triangulate(config_path3d, video_dir, videotype='.mp4', gputouse=0, filterpredictions=True)
```

### create labeled 3d videos
The folder `videos_dir` should contain the newly created.h5 file for the videos you analyzed.
```python
deeplabcut.create_labeled_video_3d(config_path3d, [videos_dir], videotype='.mp4', trailpoints=10, view=[0,270])
```
