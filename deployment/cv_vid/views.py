from django.shortcuts import render, redirect

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def base(request):
        
    return render(request, 'cv_vid/base.html')        
     

def load_model():
    # load the model for inference 
    model = models.segmentation.fcn_resnet101(pretrained=True).eval()
    return model

def get_segmentation(input_image, model):
    #input_image = Image.open(img_file)
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    rgb = seg2rgb(output_predictions)
    rgb = np.array(rgb)
    return rgb


label_colors = np.array([(0, 0, 0),  # 0=background
              # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
              (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
              # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
              (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
              # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
              (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

def seg2rgb(preds):
    colors = label_colors
    colors = label_colors.astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    rgb = Image.fromarray(preds.byte().cpu().numpy())#.resize(preds.shape)
    rgb.putpalette(colors)
    rgb = rgb.convert('RGB')
    return rgb


def semantic_segmentation(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        video_file = fs.url(filename)

        video_file_ = settings.BASE_DIR + '/' + video_file
            
        model = load_model()
        video_output = 'output.mp4'

        clip = VideoFileClip(video_file_)
        
        def process_frame(frame):        
            #img = Image.open(img_file_)
            #img = cv2.imread(filename)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return get_segmentation(frame, model)
        
        video_output = settings.MEDIA_ROOT + '/seg_vid.mp4'
        seg_clip = clip.fl_image(process_frame)
        seg_clip.write_videofile(video_output, audio=False)


        return render(request, 'cv_vid/semantic_segmentation.html', {'original_vid': video_file,
                                                                     'segmented_vid': '/media/seg_vid.mp4'})
    return render(request, 'cv_vid/semantic_segmentation.html')  

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_path, threshold):
    img = Image.open(img_path) # Load the image
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
    pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def object_detection(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        rect_th = 1 
        text_size = 0.2
        text_th = 1
        img_file_ = settings.BASE_DIR + '/' + img_file
        boxes, pred_cls = get_prediction(img_file_, threshold=0.8) # Get predictions
        img = cv2.imread(img_file_) # Read image with cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        #plt.imshow(img)
        #plt.show()
        for box, cls in zip(boxes, pred_cls):#range(len(boxes)):
            cv2.rectangle(img, box[0], box[1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
            cv2.putText(img,cls, box[0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
        
        
        obb_file = settings.MEDIA_ROOT + '/obb_img.png' 
        cv2.imwrite(obb_file, img)

        return render(request, 'cv/object_detection.html', {'original_img': img_file,
                                                            'obb_img': '/media/obb_img.png'})
        
    return render(request, 'cv/object_detection.html')  