import pandas as pd
import numpy as np
import sys
import yaml 
from nlpaug.augmenter.word import SynonymAug

from pathlib import Path


params = yaml.safe_load(open('params.yaml'))['augment']
stopwords = params['stopwords']
aug_max = params['aug_max']

if stopwords:
    stopwords = pd.read_fwf(stopwords, header=None).values
else:
    stopwords = None

input_dir = sys.argv[1]
output_dir = sys.argv[2]

train_txt_path = Path(input_dir) / 'text.txt'
train_labels_path = Path(input_dir) / 'labels.txt'

train_txt = pd.read_fwf(train_txt_path, header=None)
train_labels = pd.read_fwf(train_labels_path, header=None)

data_to_augment = train_txt[train_labels[0] == 1].copy()
augmented_labels = np.ones((data_to_augment.shape[0], 1))

augmenter = SynonymAug(aug_src='wordnet', lang='pol', aug_min=1, stopwords=stopwords, aug_max=aug_max)

augmented_data = data_to_augment.map(lambda x: augmenter.augment(x))

augmented_df = pd.concat([augmented_data, pd.DataFrame(augmented_labels)], axis=1)
standard_df = pd.concat([train_txt, train_labels], axis=1)

connected_df = pd.concat([standard_df, augmented_df], axis=0)

connected_df.sample(frac=1).reset_index(drop=True)
connected_df.columns = ['text', 'label']


augmented_text_path = Path(output_dir) / 'text.txt'
augmented_labels_path = Path(output_dir) / 'labels.txt'

connected_df['text'].to_csv(augmented_text_path, header=False, index=False, sep='')
connected_df['label'].to_csv(augmented_labels_path, header=False, index=False, sep='')







