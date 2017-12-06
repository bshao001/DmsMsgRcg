import os


def get_immediate_subfolders(input_dir):
    return [folder_name for folder_name in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, folder_name))]


if __name__ == '__main__':
    from shutil import copyfile
    from settings import PROJECT_ROOT

    otm_dir = os.path.join(PROJECT_ROOT, 'Data', 'OtmImages')
    box_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp', 'BoxImages')
    png_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp', 'PngImages')

    proc = 3
    if proc == 0:  # Generate samples from what were copied from OTM folders
        for sub_dir in get_immediate_subfolders(otm_dir):
            full_dir = os.path.join(otm_dir, sub_dir)
            for img_file in os.listdir(full_dir):
                full_path_name = os.path.join(full_dir, img_file)
                if os.path.isfile(full_path_name) and img_file.lower().endswith('.jpg'):
                    new_name = 's_' + sub_dir + '_' + img_file.lower()
                    dest = os.path.join(otm_dir, new_name)
                    os.rename(full_path_name, dest)
    elif proc == 1:  # Generate samples for drawing bounding boxes
        poor_dir = os.path.join(PROJECT_ROOT, 'Data', 'Temp', 'PoorImages')
        for img_file in os.listdir(poor_dir):
            full_path_name = os.path.join(poor_dir, img_file)
            if os.path.isfile(full_path_name) and img_file.lower().endswith('.jpg'):
                src = os.path.join(otm_dir, img_file[:-4] + '.jpg')
                dest = os.path.join(box_dir, img_file[:-4] + '.jpg')
                copyfile(src, dest)
    elif proc == 2:  # Delete those jpg files that have corresponding PNG files in box_dir
        cnt = 0
        for img_file in os.listdir(box_dir):
            full_path_name = os.path.join(box_dir, img_file)
            if os.path.isfile(full_path_name) and img_file.lower().endswith('.png'):
                os.rename(full_path_name, os.path.join(png_dir, img_file))
                jpg_file = os.path.join(box_dir, img_file[:-4] + '.jpg')
                if os.path.exists(jpg_file):
                    os.remove(jpg_file)
                    cnt += 1
        print("{} JPG files removed.".format(cnt))
    elif proc == 3:
        copied, skipped = 0, 0
        ant_dir = os.path.join(PROJECT_ROOT, 'Data', 'Step1', 'Training', 'NewAntImages')
        for img_file in os.listdir(png_dir):
            full_path_name = os.path.join(png_dir, img_file)
            if os.path.isfile(full_path_name) and img_file.lower().endswith('.png'):
                jpg_name = img_file[:-4] + '.jpg'
                src_file = os.path.join(otm_dir, jpg_name)
                dest_file = os.path.join(ant_dir, jpg_name)
                if os.path.exists(dest_file):
                    print("File {} already exists in the destination, skipped.".format(jpg_name))
                    skipped += 1
                else:
                    copyfile(src_file, dest_file)
                    copied += 1
        print("{} files copied, and {} files skipped.".format(copied, skipped))


