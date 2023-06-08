import pandas as pd
import numpy as np
import sys
import yaml 
from nlpaug.augmenter.word import SynonymAug

from pathlib import Path


params = yaml.safe_load(open('params.yaml'))['augment']
seed = params['seed']
stopwords = params['stopwords']
aug_max = params['aug_max']
proportion = params['proportion']

if stopwords:
    stopwords = pd.read_fwf(stopwords, header=None).values
else:
    stopwords = None

input_dir = sys.argv[1]
output_dir = sys.argv[2]

train_txt_path = Path(input_dir) / 'X_train.csv'
train_labels_path = Path(input_dir) / 'y_train.csv'

train_txt = pd.read_csv(train_txt_path, header=None)
train_labels = pd.read_csv(train_labels_path, header=None)

data_to_augment = train_txt[train_labels[0] == 1]
augmented_labels = np.ones(data_to_augment.shape[0])


augmenter = SynonymAug(aug_src='wordnet', lang='pol', aug_min=1, stopwords=stopwords, aug_max=aug_max)
augmented_data = data_to_augment[0].map(lambda x: augmenter.augment(x)[0])


augmented_df = pd.DataFrame({'text': augmented_data, 'label': augmented_labels})
standard_df = pd.DataFrame({'text': train_txt[0], 'label': train_labels[0]})


n_to_undersampling = augmented_df.shape[0] * 2 * proportion
undersampling = standard_df[standard_df['label'] == 0].sample(n_to_undersampling, random_state=seed).reset_index(drop=True)

data_to_augument_df = pd.DataFrame({'text': data_to_augment[0], 'label': np.ones(data_to_augment.shape[0])})

connected_df = pd.concat([data_to_augument_df, undersampling, augmented_df], axis=0, ignore_index=True)

connected_df.columns = ['text', 'label']

Path(output_dir).mkdir(parents=True, exist_ok=True)

connected_df['text'].to_csv(output_dir + '/X_train.csv', header=False, index=False)
connected_df['label'].to_csv(output_dir + '/y_train.csv', header=False, index=False)







