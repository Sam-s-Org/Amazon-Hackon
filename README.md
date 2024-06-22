# Visual AI & E-commerce
Welcome to this project! This repository contains code and resources for our visual search system that uses YOLO for object detection and classification. After identifying objects, they are compared with dataset of the same class to get similar products from Amazon.

## Overview
We developed a browser extension that allows user to search for products on Amazon similar to those found on the webpage they are browsing. For seamless experience, whenever the user clicks the extension, a screenshot is taken of the webpage and it is processed by the model to return similar products.
We also created a pipeline for product search using video. The video given by a user is processed using OpenCV and object detection by the YOLO model. The frames with a high probability of an object are then compared with the relevant dataset class to get the products. 

## Features
**Browser Extension**: For a seamless experience in searching for products found anywhere on the internet.
**Video Search**: Eliminates the need to take screenshots from videos and directly search with the video.

## Working/Pipeline
**Object Detection**: Uses YOLO (You Only Look Once) to detect and classify objects from images and videos.
**Embedding Creation**: Uses CLIP for creating embedding vectors of images and frames.
**Similarity Search**: Uses l2 norm for finding K-nearest embedding vectors.

## Installation
To get started, follow these steps:
1. Clone the repository:
```
git clone https://github.com/Sam-s-Org/Amazon-Hackon.git
cd Amazon-Hackon
```

2. Download the Zip files containing weights and other required items from: 
[Google Drive Link](https://drive.google.com/drive/folders/140h13DtZQCHc5Zt5CF00I3equNDAKSN7)

3. Extract the zip files into the already present /Server folder. So now the Directory should look like:\n
Amazon-Hackon\n
|_ Browser Extension\n
|_ Server\n
   |_ feature_file\n
   |_ files\n
   |_ model_weights\n

4. Create a python environment and install the required dependencies listed in requirements.txt:
```
pip install -r requirements.txt
```

## Usage
### Extension
To use the extension, follow these steps:
1. Run the server.py file after changing directory:
``` 
cd /Amazon-Hackon/Server
python server.py
```
2. Change the SERVER_URL in the /Amazon-Hackon/Browser Extension/popup/popup.js if required.

3. Load the extension folder "Browser Extension" using steps given on [this page](https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked).

4. Go to any webpage with any image of the product you want to browse.
5. Click on the extension to view the results!

# Working on Video Data
To see the model working on video data, run:
```
cd /Amazon-Hackon/Server
streamlit run streamlit_video.py
```