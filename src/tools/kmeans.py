import cv2, os, argparse
import numpy as np
from tqdm import tqdm


def main():
    
    main_dir = '../../data/deepfashion2/'
    dirs = ["train_img/", "test_img/", "val_img/"]
    
    mean_list, std_list = [], []
    for dir in dirs:
        print(main_dir + dir)
        for img_filename in tqdm(os.listdir(main_dir+dir)):
            #print(img_filename )
            img = cv2.imread(main_dir + dir + img_filename)
            img = img / 255.0
            mean, std = cv2.meanStdDev(img)
            mean_list.append(mean.reshape((3,)))
            std_list.append(std.reshape((3,)))
        
    mean_array = np.array(mean_list)
    std_array = np.array(std_list)
         
    mean = mean_array.mean(axis=0, keepdims=True)
    std = std_array.mean(axis=0, keepdims=True)
    print("mean = ", mean[0][::-1])
    print("std = ", std[0][::-1])


if __name__ == '__main__':
    main()
