import os

labels_dir = "cardata/train/labels"  # adjust to your path
counts = []
for f in os.listdir(labels_dir):
    if f.endswith(".txt"):
        with open(os.path.join(labels_dir, f)) as fh:
            counts.append(sum(1 for line in fh if line.strip()))

from collections import Counter
print(Counter(counts))