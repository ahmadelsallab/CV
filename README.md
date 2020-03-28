# Computer Vision Demo with Django

The aim of this project is to demonstrate how to deploy ML and CV models using Django. 

I had gone through the process of deploying ML models in a Django website, on AWS EC2 instances. The process is a bit tedious and full of small details, so I thought to share my experience.

More details [here](https://docs.google.com/document/d/1TnhdRsnZlnHUEUlko6-XeMhBXONC3PQpjLp1IhExnMc/edit#).

# Layout of the tutorial

- Local/dev side
  - ML models side
  - Django framework side
  
- Server side
  - Website settings
  - Remote server setup/Apache configuration
  - Remote instance (EC2) choice
  
# CV Tasks deployed
![CV_tasks](https://ml4a.github.io/images/figures/localization-detection.png)

Since our target is just to demo the deployment, we use pre-trained models. Note that: all the steps mentioned are still applicable in case of custom trained models.
We use examples from the top two ML frameworks: Tensorflow/Keras and Pytorch 

The tasks demonstrated are (click on each to see the corresponding colab):

- [Classification](https://colab.research.google.com/drive/1DUSjbepetl8NzR1Jyffw_R7IbR3eekFk?authuser=1#scrollTo=RUAoLChNQZBN): We use pre-trained VGG model with Keras.

![cls](https://github.com/ahmadelsallab/CV/blob/master/imgs/cls.png)

- [Semantic segmentaion](https://colab.research.google.com/drive/18JS1Uks8OGut_q-SQLGtu9xbrMpNhvnA?authuser=1#scrollTo=zc27WzsQTLdk): We use pre-trained FCN model from torchvision

![SS](https://github.com/ahmadelsallab/CV/blob/master/imgs/SS.png)

- [Object detection](https://colab.research.google.com/drive/1rwS0BNfaejB7_8mEXVLlZKSpvBjaBq2I?authuser=1#scrollTo=SvfygrRHMw0F): We use pre-trained Faster R-CNN model from torchvision 

## Why Django
Django provides a full framework, to develop both backend and front end parts of a website, in Python.
Flask is easier, but requires some extra effort to add front end part.

To start a demo website, the easiest way is to follow the official Django tutorial [here](https://docs.djangoproject.com/en/3.0/intro/tutorial01/)
## Bootstrap

We use bootstrap for the styles and "look and feel"

# Important website configurations


## urls

Here we configure the routes of our website.

We have a main application route:

__deployment/urls.py__

```
from django.conf import settings
 
urlpatterns = [
    path('cv/', include('cv.urls')),
]
 


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

The last line is particularly important, since the `MEDIA_URL` is the path to the `/media/` directory, which has all the image files (input and output), that the code will use. So it's impartant to have that URL configured so that the templates can find those images when passed from the backend. In other words, the `/media/` directory is the link or shared space the links the backend and the front end.

And 3 sub-apps routes representing our 3 tasks:

__deployment/cv/urls.py__
```
from django.urls import path
 
from . import views
 
urlpatterns = [
    path('', views.base, name='base'),
    path('classification', views.classification, name='classification'),
    path('semantic_segmentation', views.semantic_segmentation, name='semantic_segmentation'),
    path('object_detection', views.object_detection, name='object_detection')
]
```

The configured `urlpatterns` will set the default landing page to the `base.html`, which will be rendered using the backed function `views.base`. This is simply rendering the `base.html` template.

The `base.html` has the main navigation bar. When the user clicks any of the tasks, the website is routed to the required backend:

```
    <ul class="navbar-nav">
      <li class="nav-item">
        <a class="nav-link" href="classification">Classification</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="semantic_segmentation">Semantic Segmentation</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="object_detection">Object Detection</a>
      </li>
    </ul>
```

Each `href` above will route to the configured url. In `cv/urls.py` we already configured which backend handler will take care of those.

In this way, this is the main _maestro_ of the app routes.

## templates
We need front end html files that handle the requests or urls above.
Those reside in the templates/cv folder.

__base.html__
This is just a fancy website theme, that all the other 3 tasks templates inherit from. It uses bootstrap for a modern look and feel.

__classification.html__

This one will have an upload file form, and output processing part that renders the uploaded image + the returned prediction:

```
{% extends 'base.html' %}
 
 
 
{% block content %}
 
<form method="post" enctype="multipart/form-data">
    {% csrf_token %}
    <input type="file" name="myfile">
    <button type="submit">Upload</button>
  </form>
 
 
 
  {% if original_img %}
  <h3>{{prediction}}</h3>
  <img src="{{ original_img }}" alt="Prediction" width="500" height="333">
 
  {% endif %}
 
 
{% endblock %}
```


__semantic_segmentation.html__
This one will have an upload file form, and output processing part that renders the uploaded image, and the segmented image that is passed from the backed:

```
{% block content %}
 
<form method="post" enctype="multipart/form-data">
    {% csrf_token %}
    <input type="file" name="myfile">
    <button type="submit">Upload</button>
</form>
 
 
 
{% if original_img %}
<img src="{{ original_img }}" alt="Original" width="500" height="333">
 
{% endif %}
 
{% if segmented_img %}
<img src="{{ segmented_img }}" alt="Segmented" width="500" height="333">
 
{% endif %}
 
 
{% endblock %}

```

__object_detection.html__

This is similar to the `semantic_segmentation.html` template

## views

__cv/views.py__

Configure the 3 tasks call backs. This where all the action happens, or the __backend__

Each of one those does the following:

- handles the `POST` request of the upload button. If not file uploaded, the basic template (without predictions) is rendered.
- passes the uploaded image file to the prediction model.
- renders back the html template with the returned prediction from the model. In cases of object detection and segmentation, those are the paths of the saved images.

```
def classification(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        pred = #make prediction
        return render(request, 'cv/classification.html', {'original_img': img_file,
                                                            'prediction': pred})
        
    return render(request, 'cv/classification.html')   
```


```
def semantic_segmentation(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        seg_file = #get_segmentation
 
        return render(request, 'cv/semantic_segmentation.html', {'original_img': img_file,
                                                                 'segmented_img': '/media/seg_img.png'})
        
    return render(request, 'cv/semantic_segmentation.html') 
```
```
def object_detection(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        
        obb_file = #detect objects
 
 
        return render(request, 'cv/object_detection.html', {'original_img': img_file,
                                                            'obb_img': '/media/obb_img.png'})
        
    return render(request, 'cv/object_detection.html') 
```




# Installation on server

```
git clone https://github.com/ahmadelsallab/CV.git
pip install -r requirements
```

Exceptions for manual installation:
- System wide packages:
```
sudo apt install python3-opencv
```
- Memory error tensorflow:
```
pip install --no-cache-dir tensorflow
```

# Deploy local
```
cd CV/deployment
python manage.py runserver
```
# Deploy on AWS EC2
## Using Django Dev server

- On the server side:
```
python manage.py runserver 0.0.0.0:8000
```

Modify your settings file to enable the remote host IP:
```
ALLOWED_HOSTS = ["server_domain_or_IP"]
```

- On the local side: connect to your remote machine in SSH: see below, the open the ip_address:8000

__What's wrong with this approach?__
You need to keep connected with SSH to the machine. If it's closed, the website is down.

## Apache and mod_wsgi
We use Apache to prevent the above problem, and have a permenant server listening to our website requests even is SSH disconnects.
- Apache = a server program to listen on server ports and serve requests
- mod_wsgi = python wrapper package to run python based websites on the server

All details can be found [here](https://www.digitalocean.com/community/tutorials/how-to-serve-django-applications-with-apache-and-mod_wsgi-on-ubuntu-16-04)
Assuming you work with Python3

- Setup packages for apache2 and mod_wsgi
```
sudo apt-get update
sudo apt-get install python3-pip apache2 libapache2-mod-wsgi-py3
```


## EC2 Instance Choice
- Create AWS account
- Log on and go to [EC2](https://console.aws.amazon.com/ec2)
- Launch new instance
- Choose Free tier (unless you want to buy strong computation, check [pricing](https://aws.amazon.com/ec2/pricing/))
- Choose Ubuntu instance
- Choose private key
- Keep the private key, otherwise you won't be able to access the instance via SSH
- Configure the security policy. Enable ALL traffic on inbound and outbound rules

## EC2 Instance Setup 

### SSH

__Command line:__

If you press "Connect" in the EC2 console you get clear instructions, make sure to choose the instance.
Doing so is just fine to ssh. However, you still to connect FTP client to download and upload files to the instance. PUTTY can be used. 

But in this way you need to do different steps for FTP and SSH.

__MobaXTerm:__

I found MobaXTerm a conventient 2-in-1 alternative. Here’s how to use to connect to your instance:
- New session
- Configure the ip=dns name and user=ubuntu
- Add private key, which you saved when creating the instance

### Apache configuration
All details can be found [here](https://www.digitalocean.com/community/tutorials/how-to-serve-django-applications-with-apache-and-mod_wsgi-on-ubuntu-16-04)

Restart Apache:
```
sudo systemctl restart apache2
```

If failed, you can check syntax using:
```
apachectl configtest
```

For any error log:
```
cat /var/log/apache2/error.log
```
