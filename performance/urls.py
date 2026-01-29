from django.urls import path
from .views import predict_performance
from .ui_views import predict_form

urlpatterns = [
    path('predict/', predict_performance),
    path('', predict_form, name='predict_form'),
    path('form/', predict_form, name='predict_form_alt'),
]

