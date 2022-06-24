from helpers import copy_images

print("Input a list with the target subclasses to extract from the parent dataset:")
sub_classes = input()

if type(sub_classes) != list:
    raise ValueError(f"The input must be a list, current input: {type(sub_classes)}")

print("Input parent folder (directory where the original dataset is located):")
parent_folder = input()

if type(parent_folder) != str:
    raise ValueError(f"Parent folder must be a str, current type is: {type(parent_folder)}")

print("Input the name of the new directory for the new dataset:")
new_subset = input()

if type(new_subset) != str:
    raise ValueError(f"New folder name must be a str, current type is: {type(new_subset)}")

datasets = ["train", "test"]

for i in datasets:
    copy_images(parent_folder=parent_folder,
                new_subset=new_subset,
                dataset=i,
                target_labels=sub_classes)
