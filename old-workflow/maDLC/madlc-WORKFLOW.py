import deeplabcut
import tkinter
from tkinter import filedialog
import os
import glob

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
    elif type == 'vfile':
        path = filedialog.askopenfilename(parent=root, initialdir=os.getcwd(), title='select video file')
    root.withdraw()
    return path

def new_project():
    print('(y/n) do you want to create a new project?')
    console = input()
    loop = True

    while loop:
        if (console == 'y') or (console == 'yes'):
            print('enter project name')
            project_name = input()
            print('enter your name')
            your_name = input()
            print('enter batch size')
            batch = input()
            print('how many frames to extract')
            numframes = input()
            print('enter videos directory')
            videos_dir = os.path.join(file_dialog('directory'))
            config_path = deeplabcut.create_new_project(project_name, your_name, [videos_dir], videotype='.mp4', copy_videos=False, multianimal=True)
            config_file = open(config_path, 'r')
            contents = config_file.read()
            starting_text = 'individuals:'
            ending_text = 'cropping: false'
            to_replace = contents[contents.find(starting_text)+len(starting_text):contents.rfind(ending_text)]
            new_contents = contents.replace(to_replace, '\n- sheep1\n- sheep2\n- sheep3\nuniquebodyparts: []\nmultianimalbodyparts:\n- head\n- shoulder\n- back\n- dock\nskeleton:\n- - head\n  - shoulder\n- - head\n  - back\n- - head\n  - dock\n- - shoulder\n  - back\n- - shoulder\n  - dock\n- - back\n  - dock\nbodyparts: MULTI!\nstart: 0\nstop: 1\nnumframes2pick: '+str(numframes)+'\n\n    # Plotting configuration\nskeleton_color: black\npcutoff: 0.6\ndotsize: 3\nalphavalue: 0.7\ncolormap: plasma\n\n    # Training,Evaluation and Analysis configuration\nTrainingFraction:\n- 0.95\niteration: 0\ndefault_net_type: resnet_50\ndefault_augmenter: multi-animal-imgaug\nsnapshotindex: -1\nbatch_size: '+str(batch)+'\n\n    # Cropping Parameters (for analysis and outlier frame detection)\n')
            config_file.close()
            config_file = open(config_path, 'w')
            config_file.write(new_contents)
            config_file.close()
            loop = False
        elif (console == 'n') or (console == 'no'):
            config_path = os.path.join(file_dialog('file'))
            videos_dir = os.path.join(file_dialog('directory'))
            loop = False
        else:
            print('try again')
            console = input()
    return config_path, videos_dir

config_path, videos_dir = new_project()

def run_workflow(config_path, videos_dir):
    loop = True
    what_to_do = '(e) extract frames\n(l) label frames\n(t) create, train and evaluate network \n(a) analyze videos and create labeled videos\n(r) extract outliers and refine labels\n(p) plot poses and analyze skeleton\n(m) merge and retrain network\n(v) add new videos\n(exit) exit'
    print(what_to_do)
    answer = input()
    while loop:
        if answer == 'e':
            deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, cluster_resizewidth=10, cluster_step=1)
            print(what_to_do)
            answer = input()
        elif answer == 'l':
            deeplabcut.label_frames(config_path)
            print(what_to_do)
            answer = input()
        elif answer == 't':
            deeplabcut.create_multianimaltraining_dataset(config_path, net_type='resnet_50')
            print('max iterations')
            iterations = input()
            deeplabcut.train_network(config_path, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0)
            deeplabcut.evaluate_network(config_path, gputouse=0, plotting=True)
            deeplabcut.evaluate_multianimal_crossvalidate(config_path, plotting=True)
            print(what_to_do)
            answer = input()
        elif answer == 'a':
            print('batch size')
            batchsize = input()
            scorername = deeplabcut.analyze_videos(config_path, [videos_dir], videotype='.mp4', gputouse=0, batchsize=batchsize, save_as_csv=False)
            #deeplabcut.create_video_with_all_detections(config_path, [video_path], DLCscorername=scorername)
            deeplabcut.convert_detections2tracklets(config_path, [videos_dir], videotype='.mp4', track_method='box')
            man, viz = deeplabcut.refine_tracklets(config_path, os.path.join(videos_dir, glob.glob(videos_dir+'\*_bx.pickle')[0]), os.path.join(file_dialog('vfile')))

            #code stops here and I have no idea why

            deeplabcut.create_labeled_video(config_path, [videos_dir], videotype='.mp4', draw_skeleton=True, track_method='box')
            print(what_to_do)
            answer = input()
        elif answer == 'r':
            deeplabcut.extract_outlier_frames(config_path, [videos_dir], videotype='.mp4', extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True, track_method='box')
            deeplabcut.refine_labels(config_path)
            print(what_to_do)
            answer = input()
        elif answer == 'p':
            deeplabcut.plot_trajectories(config_path, [videos_dir], videotype='.mp4', track_method='box')
            #deeplabcut.analyzeskeleton(config_path, [videos_dir], videotype='.mp4', track_method='box')
            print(what_to_do)
            answer = input()
        elif answer == 'm':
            print('max iterations')
            iterations = input()
            deeplabcut.merge_datasets(config_path)
            deeplabcut.create_multianimaltraining_dataset(config_path, net_type='resnet_50')
            deeplabcut.train_network(config_path, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0)
            deeplabcut.evaluate_network(config_path, gputouse=0, plotting=True)
            deeplabcut.evaluate_multianimal_crossvalidate(config_path, plotting=True)
            print(what_to_do)
            answer = input()
        elif answer == 'v':
            new_videos_dir = os.path.join(file_dialog('directory'))
            deeplabcut.add_new_videos(config_path, [new_videos_dir], copy_videos=False)
            print(what_to_do)
            answer = input()
        elif answer == 'exit':
            loop = False
        else:
            print('try again')
            answer = input()

run_workflow(config_path, videos_dir)
