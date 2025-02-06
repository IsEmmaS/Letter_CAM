import os
import numpy as np
import pandas as pd
import cv2
"""
[A_Z Handwritten Data.csv] file can 
download from kaggle which link is
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format?resource=downloa

and you save it at archive/
"""

dataset_dir = os.path.expanduser("~/ocr/archive")


def csv2img(dataset):
    for row in range(len(dataset)):
        line = dataset.iloc[row]
        label = line[0]
        save_dir = os.path.join(dataset_dir, chr(ord("A") + label))
        os.makedirs(save_dir, exist_ok=True)
        img = line[1:].values
        img_np = 255 - np.array(img.reshape((28, 28)), dtype=np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"{row}.png"), img_np)


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(dataset_dir, "A_Z Handwritten Data.csv"))
    csv2img(data)
