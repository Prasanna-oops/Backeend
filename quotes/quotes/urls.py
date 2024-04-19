from django.contrib import admin
from django.urls import path, include
from core.views import PredictView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('core/', PredictView.as_view(), name='predict'),
    path('quotes/', include('quotes.urls')),  # Include other app URLs if needed
]
