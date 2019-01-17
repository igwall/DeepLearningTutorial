import numpy as np

##### 	Fonction de prediction de la regression lineaire ####
#		Dans ce fichier vous devrez implementer le calcul de la fonction prediction
#		de la regression logistique a partir des donnees d'apprentissage et des parametres
#		du modele.

def predictLinReg(X, theta):
	N = X.shape[0]				# Nombres d'exemples d'apprentissage
   	y = np.zeros(N)

    #Faire une fonction de prediction avec une boucle dans un premier temps
	######## Code a completer ###########

	for n in range(N):
		y[n] = 0.0						#ICI TU DOIS CODER

	#####################################

	return y
