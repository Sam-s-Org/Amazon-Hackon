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

def find_similar_images(input_image,cls,processor,device,model,index,class_name,k=2):
    input_features = extract_features(input_image,processor,device,model)
    _, indices = index.search(np.array([input_features]), k)
    df = pd.read_csv(f'files/{class_name}.csv')
        
    dataset_image_url = df['image'].values.tolist()
    dataset_link = df['link'].values.tolist()
    dataset_rate = df['ratings'].values.tolist()
    dataset_actual_price = df['actual_price'].values.tolist()
    dataset_name = df['name'].values.tolist()
    similar_images = []
    for idx in indices[0]:
        similar_images.append([dataset_name[idx],dataset_image_url[idx],dataset_link[idx],dataset_rate[idx],dataset_actual_price[idx]])
    
    return similar_images

def find_key_for_value(dictionary, value):
    keys = [key for key, val in dictionary.items() if val == value]
    return keys[0] if keys else None

def main(image, yolo_model, model, processor, device,names,class_name):
    shutil.rmtree('cropped_objects', ignore_errors=True)
    cls = find_key_for_value(names, class_name)
    ans = []
    with open(f'feature_file/{int(cls)}.pkl', 'rb') as f:
            dataset_features = pickle.load(f)
            
    dataset_features = np.array(dataset_features)
    index = faiss.IndexFlatL2(dataset_features.shape[1])
    index.add(dataset_features)
    similar = find_similar_images(image,-1,processor,device,model,index,class_name)
    for ss in similar:
        ans.append(ss)
    return ans
    