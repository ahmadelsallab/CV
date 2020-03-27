# Computer Vision Demo with Django

The aim of this project is to demonstrate how to deploy ML and CV models using Django.

![CV_tasks](https://ml4a.github.io/images/figures/localization-detection.png)

Since our target is just to demo the deployment, we use pre-trained models. Note that: all the steps mentioned are still applicable in case of custom trained models.
We use examples from the top two ML frameworks: Tensorflow/Keras and Pytorch 

The tasks demonstrated are (click on each to see the corresponding colab):

- [Classification](https://colab.research.google.com/drive/1DUSjbepetl8NzR1Jyffw_R7IbR3eekFk?authuser=1#scrollTo=RUAoLChNQZBN): We use pre-trained VGG model with Keras

- [Semantic segmentaion](https://colab.research.google.com/drive/18JS1Uks8OGut_q-SQLGtu9xbrMpNhvnA?authuser=1#scrollTo=zc27WzsQTLdk): We use pre-trained FCN model from torchvision

- [Object detection](https://colab.research.google.com/drive/1rwS0BNfaejB7_8mEXVLlZKSpvBjaBq2I?authuser=1#scrollTo=SvfygrRHMw0F): We use pre-trained Faster R-CNN model from torchvision 

## Why Django
Django provides a full framework, to develop both backend and front end parts of a website, in Python.
Flask is easier, but requires some extra effort to add front end part.

To start a demo website, the easiest way is to follow the official Django tutorial [here](https://docs.djangoproject.com/en/3.0/intro/tutorial01/)
## Bootstrap

We use bootstrap for the styles and "look and feel"


# Installation

```
git clone https://github.com/ahmadelsallab/CV.git
pip install -r requirements
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

Command line:
If you press Connect you get clear instructions, make sure to choose the instance:


Doing so is just fine to ssh. 
However, you still to connect FTP client to download and upload files to the instance.
PUTTY can be used. 
But in this way you need to do different steps for FTP and SSH

I found MobaXTerm a conventient 2-in-1 alternative.
Here’s how to use to connect to your instance:





Voila! You’re are connected to your instance!


### Packages

