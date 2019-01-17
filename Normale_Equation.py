import numpy as np

##### 	Equation normale de la regression logistique ####
#		Dans ce fichier vous devrez calculer le vecteur de parametre de la regression logistique
#       a l'aide de la resolution analytique du TP. 
#		
#		


def normalEquation(X, y):
	theta		 = np.zeros(X.shape[1])		# Historique des valeurs de la loss

	######## Code a completer ###########
	X_transpose = []]			#ICI TU DOIS CODER
	pseudo_inverse_term = []]	#ICI TU DOIS CODER
								#HINT: Utilisez la fonction np.linalg.pinv() pour inverser la matrice

	theta = []]					#ICI TU DOIS CODER
	#####################################


	return theta