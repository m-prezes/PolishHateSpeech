import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from utils import preprocessing_text

params = yaml.safe_load(open("params.yaml"))["prepare"]
split = params["val_split"]
seed = params["seed"]


train_texts = pd.read_fwf(Path(sys.argv[1]), header=None)
train_labels = pd.read_fwf(Path(sys.argv[2]), header=None)


X_train, X_val, y_train, y_val = train_test_split(
    train_texts, train_labels, test_size=split, random_state=seed, stratify=train_labels
)

X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)


X_train[0] = X_train[0].apply(lambda x: preprocessing_text(x))
X_val[0] = X_val[0].apply(lambda x: preprocessing_text(x))


Path("data/prepared/train").mkdir(parents=True, exist_ok=True)
Path("data/prepared/val").mkdir(parents=True, exist_ok=True)

X_train.to_csv("data/prepared/train/X_train.csv", index=False, header=False)
X_val.to_csv("data/prepared/val/X_val.csv", index=False, header=False)
y_train.to_csv("data/prepared/train/y_train.csv", index=False, header=False)
y_val.to_csv("data/prepared/val/y_val.csv", index=False, header=False)
