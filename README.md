# Visual AI & E-commerce
Welcome to this project! This repository contains code and resources for our visual search system that uses YOLO for object detection and classification. After identifying objects, they are compared with dataset of the same class to get similar products from Amazon.

## Overview
We developed a browser extension that allows user to search for products on Amazon similar to those found on the webpage they are browsing. For seamless experience, whenever the user clicks the extension, a screenshot is taken of the webpage and it is processed by the model to return similar products.
We also created a pipeline for product search using video. The video given by a user is processed using OpenCV and object detection by the YOLO model. The frames with a high probability of an object are then compared with the relevant dataset class to get the products. 

## Features
**Browser Extension**: For a seamless experience in searching for products found anywhere on the internet.
**Video Search**: Eliminates the need to take screenshots from videos and directly search with the video.

## Working/PipeLine
**Object Detection**: Uses YOLO (You Only Look Once) to detect and classify objects from images and videos.
**Embedding Creation**: Uses CLIP for creating embedding vectors of images and frames.
**Similarity Search**: Uses l2 norm for finding K-nearest embedding vectors.

## Installation
To get started, follow these steps:
1. Clone the repository:
 ```
 git clone https://github.com/Sam-s-Org/Amazon-Hackon.git
 ```

2. 