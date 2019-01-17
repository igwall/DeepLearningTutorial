import numpy as np
from predictLinReg import predictLinReg as f

##### 	Fonction de cout de la regression lineaire ####
#		Dans ce fichier vous devrez implementer le calcul de la fonction de cout
#		de la regression lineaire a partir des donnees d'apprentissage et des parametres
#		du modele.

def computeLoss(X, y, theta):
	loss = 0.0					# Initialisation de la valeur a retourner
	N = X.shape[0]				# Nombres d'exemples d'apprentissage

	######## Code a completer ###########
	for n in range(N): 
		loss += 0.0   #ICI TU DOIS CODER
	
	#####################################


	return loss 