import glob
from collections import Counter
train_label_files = glob.glob("C:\\Users\\tobia\\.cache\\kagglehub\\datasets\\pkdarabi\\cardetection\\versions\\5\\car\\train\\labels\\*.txt")
test_label_files = glob.glob("C:\\Users\\tobia\\.cache\\kagglehub\\datasets\\pkdarabi\\cardetection\\versions\\5\\car\\test\\labels\\*.txt")
valid_label_files = glob.glob("C:\\Users\\tobia\\.cache\\kagglehub\\datasets\\pkdarabi\\cardetection\\versions\\5\\car\\valid\\labels\\*.txt")
counter = Counter()
all_label_files = train_label_files + test_label_files + valid_label_files

for file in all_label_files:
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split( )
            class_id = int(parts[0])  # brug int() her
            counter[class_id] = counter.get(class_id, 0) + 1
print(counter)