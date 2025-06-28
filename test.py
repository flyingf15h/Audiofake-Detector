from datasets import load_dataset

ds = load_dataset("Hemg/Deepfake-Audio-Dataset")["train"]

# Split into train and test (80%/20%)
split_ds = ds.train_test_split(test_size=0.2)
train_ds = split_ds["train"]
test_ds = split_ds["test"]