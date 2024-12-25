# emotions/urls.py

from django.urls import path
from .views import EmotionRecognitionAPIView

urlpatterns = [
    path('predict_emotions/', EmotionRecognitionAPIView.as_view(), name='predict_emotions'),
    # Add other paths if needed
]
