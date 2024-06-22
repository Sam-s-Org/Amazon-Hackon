from flask import Flask,request,jsonify
from flask_cors import CORS
import torch
import base64
from model import main
from transformers import CLIPProcessor, CLIPModel
import cv2


yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model_weights/best.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

app=Flask(__name__)
CORS(app)

@app.route('/search',methods=['POST'])
def search():
    a = request.json["image"]
    # print(a)
    decoded_bytes = base64.b64decode(a)
    with open("saved/img.jpg", "wb") as f:
        f.write(decoded_bytes)
    image = cv2.imread('saved/img.jpg')
    similar = main(image, yolo_model, model, processor, device)
    
    # just a for loop to remove any nan value if present 
    # because nan values cause errors in jsonification of the data
    for i, el in enumerate(similar):
        x = []
        for el2 in el:
            if type(el2) is float:
                x.append('-1')
            else:
                x.append(el2)
        similar[i] = tuple(x)
    
    response={"matches": similar}
    return jsonify(response)

if __name__=="__main__":
    app.run(port=5001)
    print(f"Server quiting...")