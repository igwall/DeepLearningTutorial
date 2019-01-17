import numpy as np
from computeLossRegLog import computeLoss
from predictLogReg import predictLogReg as f

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
		gradient = 0.0 						# ICI TU DOIS CODER

		#Mise a jour des parametres
		theta = theta - 0.0     

		#Calcul de la fonction de cout et remplissage de l'historique des valeurs
		cost_history[iter] = computeLoss(X, y, theta)   	
		print("Valeur de la fonction de cout a l'iteration %d, J = %f" % (iter, cost_history[iter]))
	#####################################



	return theta, cost_history
