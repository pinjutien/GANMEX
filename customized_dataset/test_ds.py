import tensorflow_datasets as tfds
import a2o
ds = tfds.load("a2o", split="train")
ds = ds.take(5)
for example in ds:
    # print(example)
    image = example["image"]
    label = example["label"]
    print(image.shape, label)
