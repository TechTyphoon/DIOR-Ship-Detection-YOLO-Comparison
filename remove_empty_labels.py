# Remove all empty label files in train and val folders for YOLOv5
import os

def remove_empty_files(folder):
    removed = 0
    for fname in os.listdir(folder):
        if fname.endswith('.txt'):
            fpath = os.path.join(folder, fname)
            if os.path.getsize(fpath) == 0:
                os.remove(fpath)
                print(f"Removed empty file: {fpath}")
                removed += 1
    return removed

base_dir = 'datasets/labels'
total_removed = 0
for subfolder in ['train', 'val']:
    folder_path = os.path.join(base_dir, subfolder)
    if os.path.exists(folder_path):
        total_removed += remove_empty_files(folder_path)
    else:
        print(f"Folder not found: {folder_path}")
print(f"Total empty label files removed: {total_removed}")
