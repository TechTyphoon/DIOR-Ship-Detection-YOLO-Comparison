# Regenerate YOLO label files for 'ship' class (class 0) from sample_labels.csv
import os
import pandas as pd

csv_path = 'data/sample_labels.csv'
train_img_dir = 'datasets/images/train/'
val_img_dir = 'datasets/images/val/'
train_label_dir = 'datasets/labels/train/'
val_label_dir = 'datasets/labels/val/'

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Split: 80% train, 20% val
train_split = int(0.8 * len(df))
train_df = df.iloc[:train_split]
val_df = df.iloc[train_split:]

def create_label_files(df, img_dir, label_dir):
    for _, row in df.iterrows():
        fname = row['filename']
        label = row['label']
        # Only create label file for 'ship' class (label==1)
        if label == 1:
            label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
            # Dummy YOLO annotation: class x_center y_center width height
            # Here, we use class 0 for 'ship' and dummy bbox values
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.5 0.5\n')

create_label_files(train_df, train_img_dir, train_label_dir)
create_label_files(val_df, val_img_dir, val_label_dir)
print('YOLO label files for ship class regenerated.')
