from django.urls import path
from .ui_views import predict_form

urlpatterns = [
    path('', predict_form, name='predict_form'),
]
