import os

cam1_path = os.path.join(r'C:\Users\etarter\Downloads\calib\images\cam1')
cam2_path = os.path.join(r'C:\Users\etarter\Downloads\calib\images\cam2')

cam1_list = os.listdir(cam1_path)
for i in range(len(cam1_list)):
    cam1_list[i] = cam1_list[i][5:]
cam2_list = os.listdir(cam2_path)
for i in range(len(cam2_list)):
    cam2_list[i] = cam2_list[i][5:]

for fname in cam2_list:
    if fname not in cam1_list:
        os.remove(os.path.join(r'C:\Users\etarter\Downloads\calib\images\cam2', 'cam2-'+fname))

for fname in cam1_list:
    if fname not in cam2_list:
        os.remove(os.path.join(r'C:\Users\etarter\Downloads\calib\images\cam1', 'cam1-'+fname))

unique_list = list(set(cam1_list) & set(cam2_list))
for i in range(len(unique_list)):
    if (i%8!=0):
        os.remove(os.path.join(cam1_path, 'cam1-'+unique_list[i]))
        os.remove(os.path.join(cam2_path, 'cam2-'+unique_list[i]))
