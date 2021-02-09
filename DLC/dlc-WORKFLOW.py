import deeplabcut
import tkinter
from tkinter import filedialog
import os

def file_dialog(type):
    root = tkinter.Tk()
    root.geometry('0x0+0+0')
    root.lift()
    root.focus_force()
    root.deiconify()
    root.update_idletasks()
    # root.overrideredirect(True)
    # root.mainloop()
    if type == 'file':
        path = filedialog.askopenfilename(parent=root, initialdir= os.getcwd(), title= "select config file")
    elif type == 'directory':
        path = filedialog.askdirectory(parent=root, initialdir=os.getcwd(), title='select video directory')     #add automatic video directory assignement
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
            print('enter videos directory')
            videos_dir = os.path.join(file_dialog('directory'))
            config_path = deeplabcut.create_new_project(project_name, your_name, [videos_dir],
                                                        copy_videos=False, videotype='.mp4', multianimal=False)
            config_file = open(config_path, 'r')
            contents = config_file.read()
            starting_text = 'bodyparts:'
            ending_text = 'cropping: false'
            to_replace = contents[contents.find(starting_text)+len(starting_text):contents.rfind(ending_text)]
            new_contents = contents.replace(to_replace, '\n- ear l\n- ear r\n- nose\n- top of shoulder\n- breast\n- hoof f l\n- hoof f r\n- hoof b l\n- hoof b r\n- dock\nstart: 0\nstop: 1\nnumframes2pick: 20\n\n    # Plotting configuration\nskeleton:\n- - ear l\n  - ear r\n- - ear l\n  - nose\n- - ear r\n  - nose\n- - top of shoulder\n  - breast\n- - top of shoulder\n  - dock\n- - top of shoulder\n  - hoof f l\n- - top of shoulder\n  - hoof f r\n- - dock\n  - hoof b l\n- - dock\n  - hoof b r\nskeleton_color: black\npcutoff: 0.6\ndotsize: 3\nalphavalue: 0.7\ncolormap: plasma\n\n    # Training,Evaluation and Analysis configuration\nTrainingFraction:\n- 0.95\niteration: 0\ndefault_net_type: resnet_50\ndefault_augmenter: default\nsnapshotindex: -1\nbatch_size: '+str(batch)+'\n\n    # Cropping Parameters (for analysis and outlier frame detection)\n')
            config_file.close()
            config_file = open(config_path, 'w')
            config_file.write(new_contents)
            config_file.close()
            loop = False
        elif (console == 'n') or (console == 'no'):
            config_path = os.path.join(file_dialog('file'))
            videos_dir = os.path.join(file_dialog('directory'))
            loop = False
        elif answer == 'exit':
            loop = True
        else:
            print('try again')
            console = input()
    return config_path, videos_dir

def run_workflow(config_path, videos_dir):
    loop = True
    what_to_do = '(e) extract frames\n(l) label frames\n(t) create, train and evaluate network \n(a) analyze videos and create labeled videos\n(r) extract outliers and refine labels\n(m) merge and retrain network\n(v) add new videos\n(exit) exit'
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
            deeplabcut.create_training_dataset(config_path, net_type='resnet_50', augmenter_type='imgaug')
            print('max iterations')
            iterations = input()
            deeplabcut.train_network(config_path, displayiters=10, maxiters=iterations, allow_growth=True, gputouse=0)
            deeplabcut.evaluate_network(config_path, gputouse=0, plotting=True)
            print(what_to_do)
            answer = input()
        elif answer == 'a':
            videos_dir = os.path.join(file_dialog('directory'))
            print('batch size')
            batchsize = input()
            deeplabcut.analyze_videos(config_path,[videos_dir], videotype='.mp4', gputouse=0, batchsize=batchsize, save_as_csv=False)
            deeplabcut.create_labeled_video(config_path, [videos_dir], videotype='.mp4', draw_skeleton=True)
            print(what_to_do)
            answer = input()
        elif answer == 'r':
            deeplabcut.extract_outlier_frames(config_path, [videos_dir], videotype='.mp4', extractionalgorithm='kmeans', cluster_resizewidth=10, automatic=True, cluster_color=True)
            deeplabcut.refine_labels(config_path)
            print(what_to_do)
            answer = input()
        elif answer == 'm':
            print('max iterations')
            iterations = input()
            deeplabcut.merge_datasets(config_path)
            deeplabcut.create_training_dataset(config_path, net_type='resnet_50', augmenter_type='imgaug')
            deeplabcut.train_network(config_path, displayiters=100, maxiters=iterations, allow_growth=True, gputouse=0)
            deeplabcut.evaluate_network(config_path, gputouse=0, plotting=True)
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

config_path, videos_dir = new_project()
run_workflow(config_path, videos_dir)
