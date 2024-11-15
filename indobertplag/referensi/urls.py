from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login_view'),
    path('logout/', views.logout_view, name='logout_view'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('add_reference/', views.add_reference, name='add_reference'),
    path('delete_reference/<int:reference_id>/', views.delete_reference, name='delete_reference'),
    path('recalculate_embeddings/', views.recalculate_embeddings, name='recalculate_embeddings')
]