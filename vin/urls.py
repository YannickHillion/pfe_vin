from django.conf.urls import url
from . import views

urlpatterns = [

    url(r'^$', views.index_vin, name='index_vin'),
    url(r'^predic_user_vin/(?P<id_user>\w+)/(?P<id_vin>\w+)/$', views.predic_user_vin, name='predic_user_vin'),
    url(r'^predic_user_top/(?P<id_user>\w+)/$', views.predic_user_top, name='predic_user_top'),
     url(r'^print_result/$', views.print_result, name='print_result'),

]
