from dataset_loader import HabitatDataset

dataset_path = "../habitat_dataset"

dataset = HabitatDataset(dataset_path)

print("Dataset size:", len(dataset))

image, target = dataset[0]

print("Image shape:", image.shape)

print("Target:", target)
