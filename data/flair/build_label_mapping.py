import json
import os

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

dataset_dir = "./data" # replace with the path to directory that you downloaded the dataset

with open(os.path.join(dataset_dir, "labels_and_metadata.json")) as f:
    metadata_list = json.load(f)

# Counter for image label statistics
image_label_counter = []
image_fine_grained_counter = []

# Counters for label statistics
label_counter = Counter()
fine_grained_label_counter = Counter()

for metadata in metadata_list:
    image_label_counter.append(len(metadata["labels"]))
    image_fine_grained_counter.append(len(metadata["fine_grained_labels"]))
    # Increment count for overall label distribution
    label_counter.update(metadata["labels"])
    fine_grained_label_counter.update(metadata["fine_grained_labels"])

labels = list(label_counter.keys())
fg_labels = list(fine_grained_label_counter.keys())

label_mapping = {}
fg_label_mapping = {}

for idx, l in enumerate(labels):
    label_mapping[l] = [0] * len(labels)
    label_mapping[l][idx] = 1
print(label_mapping)

for idx, l in enumerate(fg_labels):
    fg_label_mapping[l] = [0] * len(fg_labels)
    fg_label_mapping[l][idx] = 1

with open('./label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent=4)

with open('./fine_grained_label_mapping.json', 'w') as f:
    json.dump(fg_label_mapping, f, indent=4)