# Imports
import numpy as np
import pandas as pd
import os, re, cv2, pydicom, warnings

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut

from engine import train_one_epoch, evaluate
import utils
import transforms as T

# Hyperparameters
# test_set_length = 40 		 # Test set (number of images)
val_set_length = 3000
train_batch_size = 4	 # Train batch size
# test_batch_size = 16    		 # Test batch size
val_batch_size = 4
num_classes = 14+1        		 # Number of classes
# learning_rate = 0.005  		 # Learning rate
learning_rate = 0.0001
num_epochs = 100    	     # Number of epochs
output_dir = "/data1/geun_19/G-ff/py-faster_rcnn/weight/"   # Output directory to save the model
base_dir = "/data1/geun_19/G-ff/py-faster_rcnn/data/"
train_dicom_dir = base_dir + "train/"
test_dicom_dir =  base_dir + "test/"
val_dicom_dir =  base_dir + "val/"

train_df_dir = base_dir +"f_train.csv"
valid_df_dir = base_dir +"f_val.csv"

train_df = pd.read_csv(train_df_dir)
valid_df = pd.read_csv(valid_df_dir)

def create_label_txt(path_to_csv):

	data = pd.read_csv(path_to_csv)
	labels = data['class_name'].unique()

	labels_dict = {}

	# Creat dictionary from array
	for index, label in enumerate(labels):
		labels_dict.__setitem__(index, label)

	# We need to create labels.txt and write labels dictionary into it
	# with open('/data1/geun_19/pytorch_custom_object_detection/data/labels.txt', 'w') as f:
	# 	f.write(str(labels_dict))

	return labels_dict	


def parse_one_annot(path, filename, labels_dict):

	data = pd.read_csv(path)

	class_names = data['class_name'].unique()
	# classes_df = data[data["filename"] == filename]["class"]
	classes_df = data[data["image_id"]+".dicom" == filename]["class_name"]
	classes_array = classes_df.to_numpy()
	
	# boxes_df = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]]
	boxes_df = data[data["image_id"]+".dicom" == filename][["x_min", "y_min", "x_max", "y_max"]]
	boxes_array = boxes_df.to_numpy()
	
	classes = []
	for key, value in labels_dict.items():
		for i in classes_array:
			if i == value:
				classes.append(key)

	# Convert list to tuple
	classes = tuple(classes)

	return boxes_array, classes

class VinBigDataset(torch.utils.data.Dataset): #Class to load Training Data
    
    def __init__(self, dataset_dir, csv_file, labels_dict, transforms = None):
        
        self.dataset_dir = dataset_dir
        self.csv_file = csv_file
        self.transforms = transforms
        self.labels_dict = labels_dict
        self.image_names = [file for file in sorted(os.listdir(os.path.join(dataset_dir))) if file.endswith('.dicom')]


    def __getitem__(self, index):
            
        image_path = os.path.join(self.dataset_dir, self.image_names[index])

        image = pydicom.dcmread(image_path)

        image = image.pixel_array
        
        if "PhotometricInterpretation" in image:
            if image.PhotometricInterpretation == "MONOCHROME1":
                image = np.amax(image) - image

        intercept = image.RescaleIntercept if "RescaleIntercept" in image else 0.0
        slope = image.RescaleSlope if "RescaleSlope" in image else 1.0

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
            
        image += np.int16(intercept)        

        image = np.stack([image, image, image])
        image = image.astype('float32')
        image = image - image.min()
        image = image / image.max()
        image = image * 255.0
        image = image.transpose(1,2,0)

        box_array, classes = parse_one_annot(self.csv_file, self.image_names[index], self.labels_dict)
        boxes = torch.as_tensor(box_array, dtype = torch.float32)
        labels = torch.tensor(classes, dtype=torch.int64)    
        
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def __len__(self):

        return len(self.image_names)
    
    
def get_model(num_classes):

	# Load an pre-trained object detectin model (in this case faster-rcnn)
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

	# Number of input features
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# Replace the pre-trained head with a new head
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model


def get_transforms(train):

	transforms = []

	# Convert numpy image to PyTorch Tensor
	transforms.append(T.ToTensor())

	if train:
		# Data augmentation
		transforms.append(T.RandomHorizontalFlip(0.5))

	return T.Compose(transforms)


if __name__ == '__main__':

    # Setting up the device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    labels_dict = create_label_txt(train_df_dir)

    train_dataset = VinBigDataset(dataset_dir = train_dicom_dir, csv_file = train_df_dir, labels_dict = labels_dict, transforms = get_transforms(train = True))
    valid_dataset = VinBigDataset(dataset_dir = val_dicom_dir, csv_file = valid_df_dir, labels_dict = labels_dict, transforms = get_transforms(train = True))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True,
        num_workers = 4, collate_fn = utils.collate_fn)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = val_batch_size, shuffle = False,
        num_workers = 4, collate_fn = utils.collate_fn)

    print(f"We have: {len(train_dataset)+len(valid_dataset)} images in the dataset, {len(train_dataset)} are training images and {len(valid_dataset)} are validation images")


    # Get the model using helper function
    model = get_model(num_classes)
    model.to(device = device)

    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 0.0005)

	# Learning rate scheduler decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)


    for epoch in range(num_epochs):
		
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq = 10)
        lr_scheduler.step()
        # Evaluate on the test dataset
        evaluate(model, valid_data_loader, device = device)

        torch.save(model.state_dict(), output_dir+'faster_rcnn-bb_resnet50_'+str(epoch+1)+'.pth')


	# if not os.path.exists(output_dir):
	# 	os.mkdir(output_dir)

	# Save the model state	
	# torch.save(model.state_dict(), output_dir+'faster_rcnn-bb_resnet50_'+str(epoch+1)+'.pth')