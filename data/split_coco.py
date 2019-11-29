import json
import numpy as np

with open("data/widgets/annotations/all.json") as f:
    all_annotations = json.load(f)

samples_count = len(all_annotations["annotations"])
validation_ids = np.random.choice(samples_count, size=62, replace=False)
valid_dict = all_annotations.copy()
valid_dict["annotations"] = [valid_dict["annotations"][i] for i in validation_ids]

train_dict = all_annotations.copy()
train_dict["annotations"] = [train_dict["annotations"][i] for i in range(samples_count) if i not in validation_ids]

with open("data/widgets/annotations/widgets-train.json", "w") as f:
    json.dump(train_dict, f)
with open("data/widgets/annotations/widgets-val.json", "w") as f:
    json.dump(valid_dict, f)
