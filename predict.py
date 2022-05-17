import torch
import numpy as np
import pandas as pd
import glob
import os, re, cv2, pydicom, warnings
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# saved_model = "saved_model"  # Output directory of the save the model
# filename = "IMG_2558.JPG"    # Image filename
# img_path = "/data1/geun_19/G-ff/py-faster_rcnn/data/val/" + filename
test_path = "/data1/geun_19/G-ff/py-faster_rcnn/data/val/"
img_path = glob.glob(test_path+'/*.dicom')[0]
num_classes = 14+1
with open('/data1/geun_19/pytorch_custom_object_detection/data/labels.txt', 'r') as f:
	string = f.read()
	labels_dict = eval(string)

def get_model(num_classes):

	# Load an pre-trained object detectin model (in this case faster-rcnn)
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

	# Number of input features
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# Replace the pre-trained head with a new head
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model

# image = cv2.imread(img_path)
image = pydicom.dcmread(img_path)
image = image.pixel_array
# if "PhotometricInterpretation" in image:
# 	if image.PhotometricInterpretation == "MONOCHROME1":
# 		image = np.amax(image) - image

# intercept = image.RescaleIntercept if "RescaleIntercept" in image else 0.0
# slope = image.RescaleSlope if "RescaleSlope" in image else 1.0

# if slope != 1:
# 	image = slope * image.astype(np.float64)
# 	image = image.astype(np.int16)
            
# image += np.int16(intercept) 
# image = np.stack([image, image, image])
image = image.astype('float32')
image = image - image.min()
image = image / image.max()
image = image * 255.0
# image = image.transpose(1,2,0)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = torchvision.transforms.ToTensor()(img)

loaded_model = get_model(num_classes)
weight_path = "/data1/geun_19/G-ff/py-faster_rcnn/weight/faster_rcnn-bb_resnet50_35.pth"
# loaded_model.load_state_dict(torch.load(os.path.join(saved_model, 'model'), map_location = 'cpu'))
loaded_model.load_state_dict(torch.load(weight_path, map_location = 'cpu'))

loaded_model.eval()
with torch.no_grad():
	prediction = loaded_model([image])

for element in range(len(prediction[0]['boxes'])):
	x, y, w, h = prediction[0]['boxes'][element].numpy().astype(int)
	score = np.round(prediction[0]['scores'][element].numpy(), decimals = 3)
	label_index = prediction[0]['labels'][element].numpy()
	label = labels_dict[int(label_index)]
	
	if score > 0.7:
		cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 2)
		text = str(label) + " " + str(score)
		cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					 (255, 255, 255), 1)

# cv2.imshow("Predictions", image)
# cv2.waitKey(0)
cv2.imwrite("/data1/geun_19/pytorch_custom_object_detection/results/test_images/"+"1.png", image)