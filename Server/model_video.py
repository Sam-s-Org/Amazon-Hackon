import shutil
from PIL import Image
import torch
import os
import cv2
import os
import numpy as np
import pickle
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from sklearn.neighbors import NearestNeighbors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

def extract_features(image,processor,device,model):
    image = Image.open(image)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy().reshape(-1)
    return image_features

def find_similar_images(input_image,cls,neighbors,processor,device,model,names):
    input_features = extract_features(input_image,processor,device,model)
    req_class = ['Bags & Luggage','All Electronics','Watches']
    _ , indices = neighbors.kneighbors([input_features])
    if( cls ==-1):
        df = pd.read_csv('files/general.csv')
    else:
        df = pd.read_csv(f'files/{req_class[cls]}.csv')
        
    dataset_image_url = df['image'].values.tolist()
    dataset_link = df['link'].values.tolist()
    dataset_rate = df['ratings'].values.tolist()
    dataset_actual_price = df['actual_price'].values.tolist()
    dataset_name = df['name'].values.tolist()
    similar_images = []
    for idx in indices[0]:
        similar_images.append([dataset_name[idx],dataset_image_url[idx],dataset_link[idx],dataset_rate[idx],dataset_actual_price[idx]])
    
    return similar_images

def general_search(cropped_img_path,processor,device,model,names):
    with open('feature_file/general.pkl', 'rb') as f:
        dataset_features = pickle.load(f)
        
    x = []
    for i in dataset_features:
        for j in i:
            x.append(j)
            
    dataset_features = np.array(x)
    neighbors = NearestNeighbors(n_neighbors=3, algorithm='brute',metric='euclidean').fit(dataset_features)
    
    similar = find_similar_images(cropped_img_path,-1,neighbors,processor,device,model,names)
    return similar

def main(image, yolo_model):
    shutil.rmtree('cropped_objects_2', ignore_errors=True)
    names = yolo_model.names
    img = cv2.imread(image)
    results = yolo_model(img)
    bboxes = results.xyxy[0].cpu().numpy()
    if(len(bboxes)==0):
        os.makedirs('cropped_objects_2/general', exist_ok=True)
        cv2.imwrite('cropped_objects_2/general/general.jpg', img)
        image = 'cropped_objects_2/general/general.jpg'
        return general_search(image,processor,device,model,names)
    
    gen_c = 0 
    ans=[]
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, _ , cls = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_img = img[y1:y2, x1:x2]
        det_name = names[int(cls)]
        names_cat1 = ['backpack','handbag','suitcase']
        names_cat2 = ['laptop','mouse','remote','keyboard','cell phone']
        names_cat3 = ['clock']
        if det_name in names_cat1:
            gen_c+=0
            j=0
        elif det_name in names_cat2:
            gen_c+=1
            j=1
        elif det_name in names_cat3:
            j=2
            gen_c+=2
        else:
            gen_c+=-1
            j=-1
        if j!=-1:
            os.makedirs(f'cropped_objects_2/{det_name.lower()}', exist_ok=True)
            cropped_img_path = f'cropped_objects_2/{det_name.lower()}/cropped_img_{i}.jpg'
            cv2.imwrite(cropped_img_path,cropped_img)
            
            cv2.imwrite(cropped_img_path, cropped_img)
            with open(f'feature_file/{j}.pkl', 'rb') as f:
                dataset_features = pickle.load(f)
                
            dataset_features = np.array(dataset_features)
            neighbors = NearestNeighbors(n_neighbors=3, algorithm='brute',metric='euclidean').fit(dataset_features)
            
            similar = find_similar_images(cropped_img_path,j,neighbors,processor,device,model,names)
            for ss in similar:
                ans.append(ss)
            
    if gen_c == -len(bboxes):
        os.makedirs('cropped_objects_2/general', exist_ok=True)
        cv2.imwrite('cropped_objects_2/general/general.jpg', img)
        image = 'cropped_objects_2/general/general.jpg'
        return general_search(image,processor,device,model,names)
    else:
        return ans
            
    
    
    





    
    

    
    



