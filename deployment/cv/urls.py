"""deployment URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('upload', views.upload, name='upload'),
    path('classify_api', views.classify_api, name='classify_api'),
	path('semantic_seg_api', views.semantic_seg_api, name='semantic_seg_api'),
    path('object_det_api', views.object_det_api, name='object_det_api')
]