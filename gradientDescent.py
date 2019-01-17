import numpy as np
from computeLoss import computeLoss
from predictLinReg import predictLinReg as f

##### 	Descente de gradient de la fonction de cout de la regression logistique ####
#		Dans ce fichier vous devrez implementer la descente du gradient de la fonction de cout
#		de la regression logistique a partir des donnees d'apprentissage et des parametres
#		du modele.
#
#		Votre code devra egalement retourner l'historique des valeurs de la fonction de 
#		cout pour chacune des iterations.

def gradientDescent(X, y, theta, learning_rate, num_iters):
	N			 = X.shape[0]				# Nombres d'exemples d'apprentissage
	cost_history = np.zeros(num_iters)		# Historique des valeurs de la loss

	######## Code a completer ###########

	for iter in range(num_iters):
		#Initialisation du vecteur de gradient
		gradient = np.zeros(theta.shape)	
		
		for n in range(N):
			gradient[0] += 0.0				#ICI TU DOIS CODER
			gradient[1] += 0.0				#ICI TU DOIS CODER
		
		

		#Mise a jour des parametres
		theta[0] = theta[0] - 0.0      #ICI TU DOIS CODER
		theta[1] = theta[1] - 0.0		#ICI TU DOIS CODER

		#Calcul de la fonction de cout et remplissage de l'historique des valeurs
		cost_history[iter] = 0.0	  		#ICI TU DOIS CODER
		print("Valeur de la fonction de cout a l'iteration %d, J = %f" % (iter, cost_history[iter]))
	#####################################



	return theta, cost_history