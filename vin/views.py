from django.shortcuts import render
from .tools import *
# Create your views here.

def index_vin(request):

	context = locals()
	template = 'index_vin.html'
	return render(request, template, context)


def predic_user_vin(request,id_user,id_vin):
	liste_result = predic_user_vin_tool(id_user,id_vin)

	context = locals()
	template = 'predic_user_vin.html'
	return render(request, template, context)


def predic_user_top(request,id_user):
	liste_result = predic_user_top_tool(id_user)
	vin1 = (liste_result[0][0],str(liste_result[0][1])[:6])
	vin2 = (liste_result[1][0],str(liste_result[1][1])[:6])
	vin3 = (liste_result[2][0],str(liste_result[2][1])[:6])
	vin4 = (liste_result[3][0],str(liste_result[3][1])[:6])
	vin5 = (liste_result[4][0],str(liste_result[4][1])[:6])


	context = locals()
	template = 'predic_user_top.html'
	return render(request, template, context)


def print_result(request):

	context = locals()
	template = 'print_result.html'
	return render(request, template, context)
