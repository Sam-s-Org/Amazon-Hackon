from PIL import Image
import torch
import os
import cv2
import os
import numpy as np
import pickle
import pandas as pd
import shutil
import faiss


names = ['Bags & Luggage',
 'Casual Shoes',
 'Formal Shoes',
 'Watches',
 'Shoes',
 'Jeans',
 "Men's Fashion",
 "Kids' Shoes",
 'All Electronics']

def extract_features(image,processor,device,model):
    image = Image.open(image)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy().reshape(-1)
    return image_features

def find_similar_images(input_image,cls,processor,device,model,index,k=3):
    input_features = extract_features(input_image,processor,device,model)
    _, indices = index.search(np.array([input_features]), k)
    if( cls ==-1):
        df = pd.read_csv('files/general.csv')
    else:
        df = pd.read_csv(f'files/{names[int(cls)]}.csv')
        
    dataset_image_url = df['image'].values.tolist()
    dataset_link = df['link'].values.tolist()
    dataset_rate = df['ratings'].values.tolist()
    dataset_actual_price = df['actual_price'].values.tolist()
    dataset_name = df['name'].values.tolist()
    similar_images = []
    for idx in indices[0]:
        similar_images.append([dataset_name[idx],dataset_image_url[idx],dataset_link[idx],dataset_rate[idx],dataset_actual_price[idx]])
    
    return similar_images

def main(image, yolo_model, model, processor, device):
    shutil.rmtree('cropped_objects', ignore_errors=True)
    img = image
    results = yolo_model(img)
    
    bboxes = results.xyxy[0].cpu().numpy()
    if(len(bboxes)==0):
        with open('feature_file/general.pkl', 'rb') as f:
            dataset_features = pickle.load(f)
            
        x = []
        for i in dataset_features:
            for j in i:
                x.append(j)
                
        dataset_features = np.array(x)
        index = faiss.IndexFlatL2(dataset_features.shape[1])
        index.add(dataset_features)
        os.makedirs('cropped_objects/general', exist_ok=True)
        cv2.imwrite('cropped_objects/general/no_objects.jpg',img)
        image = 'cropped_objects/general/no_objects.jpg'
        similar = find_similar_images(image,-1,processor,device,model,index)
        return similar 
        
    ans = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, _ , cls = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_img = img[y1:y2, x1:x2]
        os.makedirs(f'cropped_objects/{names[int(cls)].lower()}', exist_ok=True)
        cropped_img_path = f'cropped_objects/{names[int(cls)].lower()}/cropped_img_{i}.jpg'
        cv2.imwrite(cropped_img_path,cropped_img)
        
        with open(f'feature_file/{int(cls)}.pkl', 'rb') as f:
            dataset_features = pickle.load(f)
            
        dataset_features = np.array(dataset_features)
        index = faiss.IndexFlatL2(dataset_features.shape[1])
        index.add(dataset_features)
        
        similar = find_similar_images(cropped_img_path,cls,processor,device,model,index)
        for ss in similar:
            ans.append(ss)
        
    return ans
    
    
    





    
    

    
    



