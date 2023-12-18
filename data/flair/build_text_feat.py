from cProfile import label
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

# print(fine_grained_label_counter)
ans = {key: [] for key in labels}


with open(os.path.join(dataset_dir, "label_relationship.txt")) as f2:
    label_rel = f2.readlines()


for val in label_rel:
    tmp = val.split('->')
    fg, cg = list(map(str.strip, tmp))
    ans[cg].append((fine_grained_label_counter[fg], fg))

for item in ans:
    ans[item].sort(reverse=True)

for item in ans:
    for i, val in enumerate(ans[item]):
        ans[item][i] = val[1]
print(ans)

with open('./label_map_for_text.json', 'w') as fout:
    json.dump(ans, fout, indent=4)

## output: {key=coarse labels, value=list of fg labels}
