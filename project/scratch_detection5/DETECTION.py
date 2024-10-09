from pathlib import Path
import numpy as np
import os, shutil
import matplotlib.pyplot as plt

from PIL import Image

from tqdm.auto import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score




transform = transforms.Compose([ #TRANSFORMS THE IMAGE FOR FEATURE EXTRACTION
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



class resnet_feature_extractor(torch.nn.Module):
    def __init__(self):
        """This class extracts the feature maps from a pretrained Resnet model."""
        super(resnet_feature_extractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        

        # Hook to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)            
        self.model.layer3[-1].register_forward_hook(hook) 

    def forward(self, input):

        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]         # Feature map sizes h, w
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)            # Merge the resized feature maps
        patch = patch.reshape(patch.shape[1], -1).T   # Craete a column tensor

        return patch



memory_bank =[]

folder_path = Path(r'C:\Users\MILAN\Desktop\tfpro\machine_learning_projects\anomaly_demo\carpet\carpet\train\good')

for pth in tqdm(folder_path.iterdir(),leave=False):
    
    with torch.no_grad():
        data = transform(Image.open(pth)).unsqueeze(0)
        features = backbone(data)
        memory_bank.append(features.cpu().detach())

memory_bank = torch.cat(memory_bank,dim=0)



selected_indices = np.random.choice(len(memory_bank), size=len(memory_bank)//100, replace=False)
memory_bank = memory_bank[selected_indices]



y_score=[]
folder_path = Path(r'C:\Users\MILAN\Desktop\tfpro\machine_learning_projects\anomaly_demo\carpet\carpet\train\good')

for pth in tqdm(folder_path.iterdir(),leave=False):
    data = transform(Image.open(pth)).unsqueeze(0)
    with torch.no_grad():
        features = backbone(data)
    distances = torch.cdist(features, memory_bank, p=2.0)
    dist_score, dist_score_idxs = torch.min(distances, dim=1) 
    s_star = torch.max(dist_score)
    segm_map = dist_score.view(1, 1, 28, 28) 

    y_score.append(s_star.cpu().numpy())



best_threshold = np.mean(y_score) + 2 * np.std(y_score)



y_score = []
y_true=[]

#for classes in ['color','good','cut','hole','metal_contamination','thread']:
for classes in ['a','good']:
    folder_path = Path(r'C:\Users\MILAN\Desktop\tfpro\machine_learning_projects\anomaly_demo\carpet\carpet\test\{}'.format(classes))

    for pth in tqdm(folder_path.iterdir(),leave=False):

        class_label = pth.parts[-2]
        with torch.no_grad():
            test_image = transform(Image.open(pth)).unsqueeze(0)
            features = backbone(test_image)

        distances = torch.cdist(features, memory_bank, p=2.0)
        dist_score, dist_score_idxs = torch.min(distances, dim=1) 
        s_star = torch.max(dist_score)
        segm_map = dist_score.view(1, 1, 28, 28) 

        y_score.append(s_star.cpu().numpy())
        y_true.append(0 if class_label == 'good' else 1)



fpr, tpr, thresholds = roc_curve(y_true, y_score)
f1_scores = [f1_score(y_true, y_score >= threshold) for threshold in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]



import cv2
from datetime import datetime
from skimage.feature.peak import peak_local_max 
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480) #4128
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) #3096
while cap.isOpened():
    
    ret, frame = cap.read()
    
    #print(frame.shape)
    # Make detections 
    
    PIL_image = Image.fromarray(np.uint8(frame)).convert('RGB')
    
    #PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
    test_image = transform(PIL_image).unsqueeze(0)
    
    
    with torch.no_grad():
        features = backbone(test_image)
    
    # Forward pass
    #start = datetime.now()
    distances = torch.cdist(features, memory_bank, p=2.0)
    #print(datetime.now() - start) 
    dist_score, dist_score_idxs = torch.min(distances, dim=1) 
    
    s_star = torch.max(dist_score)
    segm_map = dist_score.view(1, 1, 28, 28) 
      
    segm_map = torch.nn.functional.interpolate(     # Upscale by bi-linaer interpolation to match the original input resolution
                segm_map,
                size=(480, 640), #(480, 640)
                mode='bilinear'
            ).cpu().squeeze().numpy()
        
    y_score_image = s_star.cpu().numpy()
    
    y_pred_image = 1*(y_score_image >= best_threshold)
    #heatmap = cv2.applyColorMap(segm_map, cv2.COLORMAP_HOT)
    class_label = ['OK','NOK']
    #print(class_label[y_pred_image])
    heat_map = segm_map

    # Normalize the heat_map data to the range [0, 255]
    min_val = best_threshold
    max_val = best_threshold * 2
    heat_map_normalized = np.clip(heat_map, min_val, max_val)
    heat_map_normalized = 255 * (heat_map_normalized - min_val) / (max_val - min_val)
    heat_map_normalized = heat_map_normalized.astype(np.uint8)

    # Apply the 'jet' colormap using OpenCV
    colormap = cv2.COLORMAP_JET
    heat_map_colored = cv2.applyColorMap(heat_map_normalized, colormap)
    
    #heatmap = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    #_,heatmap = cv2.threshold(heatmap,best_threshold,best_threshold*2,cv2.THRESH_BINARY)
    #heat_map = heat_map.all() > best_threshold and heat_map.all() < best_threshold * 2
    #print(heat_map)
    #results = model(frame)
    #cv2.imshow('frame',frame)
    #cv2.imshow('YOLO', np.squeeze(results.render()))
    #resized_heat = cv2.resize(frame,(224,224))
    #super_imposed_img = cv2.addWeighted(resized_heat, 0.1, frame, 0.7, 0)
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.7, min_distance=50) 
    #print(peak_coords)
    for i in range(0,peak_coords.shape[0]):
        #print(i)
        y = peak_coords[i,0]
        x = peak_coords[i,1]
        #print(x,y)
        top_left = (int(x - 50), int(y - 50))
        bottom_right = (int(x + 50), int(y + 50))
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        #cv2.rectangle(heat_map_colored, (x - 25, y - 25), (50,50), (0, 255, 0), 2)   
    #results = model(frame)
    #cv2.imshow('YOLO', frame)
    cv2.imshow('heat',frame)
    #cv2.imshow('heat_map',super_imposed_img)
    cv2.imshow('ff',heat_map_colored)
    #plt.imshow(segm_map.squeeze(), cmap='jet')
    #cv2.imshow('heatmap', heatmap)
    #plt.imshow((heat_map > best_threshold*1.25), cmap='gray')
    #plt.imshow(heat_map, cmap='jet',vmin=best_threshold, vmax=best_threshold*2) 
    # plt.title(f'Anomaly score: {y_score_image / best_threshold:0.4f} || {class_label[y_pred_image]}')
    #cv2.imshow('heat_map',heatmap.squeeze())
    #cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



