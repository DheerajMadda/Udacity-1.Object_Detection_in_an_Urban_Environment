import argparse
import glob
import os
import random
import shutil
import numpy as np

# from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.

    data_path = os.path.join(os.getcwd(),data_dir,'waymo','training_and_validation')
    data = random.shuffle(os.listdir(data_path))
    split_ratio = 0.9
    split_index = int(len(data) * split_ratio)
    train_data, val_data = data[:split_index], data[split_index:]

    for _file in train_data:
        shutil.move(os.path.join(data_path,_file), os.path.join(os.getcwd(),data_dir,'waymo','train'))
    for _file in val_data:
        shutil.move(os.path.join(data_path,_file), os.path.join(os.getcwd(),data_dir,'waymo','val'))



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
