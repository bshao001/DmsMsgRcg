import os

from settings import PROJECT_ROOT
from mesgclsf.datapreptools import get_immediate_subfolders

if __name__ == "__main__":
    ext_filter = ['.jpg', '.png']
    in1_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Training', 'Positive')
    in2_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step2', 'Training')

    file_list1 = []
    for img_file1 in os.listdir(in1_dir):
        full_path_name1 = os.path.join(in1_dir, img_file1)
        if os.path.isfile(full_path_name1) and img_file1.lower().endswith(tuple(ext_filter)):
            file_list1.append(img_file1)

    file_list2 = []
    for sub_dir in get_immediate_subfolders(in2_dir):
        full_dir = os.path.join(in2_dir, sub_dir)
        for sub_dir2 in get_immediate_subfolders(full_dir):
            full_dir2 = os.path.join(full_dir, sub_dir2)

            for img_file2 in os.listdir(full_dir2):
                full_path_name2 = os.path.join(full_dir2, img_file2)
                if os.path.isfile(full_path_name2) and img_file2.lower().endswith(tuple(ext_filter)):
                    file_list2.append(img_file2)

    diff21 = list(set(file_list2) - set(file_list1))
    print("Files in step2 and not in step1:")
    for f21 in diff21:
        print(f21)

    diff12 = list(set(file_list1) - set(file_list2))
    print("Files in step1 and not in step2:")
    for f12 in diff12:
        print(f12)