import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_pandas

"""
[A_Z Handwritten Data.csv] file can 
download from kaggle which link is
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format?resource=downloa

and you save it at archive/
"""
tqdm_pandas()
dataset_dir = os.path.expanduser("~/ocr/archive/")


def csv2img(dataset):
    label_count = {i: 0 for i in range(26)}

    for row in tqdm(range(len(dataset)), desc="Processing images", unit="image"):
        line = dataset.iloc[row]
        label = line.iloc[0]
        save_dir = os.path.join(dataset_dir, chr(ord("A") + label))

        if label_count[label] >= 32*4:
            continue

        os.makedirs(save_dir, exist_ok=True)
        img = line.iloc[1:].values
        img_np = 255 - np.array(img.reshape((28, 28)), dtype=np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"{row}.png"), img_np)

        label_count[label] += 1


if __name__ == "__main__":
    data = pd.read_csv(
        os.path.join(dataset_dir, "A_Z Handwritten Data.csv"), chunksize=1000
    )
    data = pd.concat(tqdm_pandas(data, desc="Reading CSV file", unit="rows"))

    csv2img(data)
