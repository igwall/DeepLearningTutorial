import numpy as np

##### 	Fonction qui ajoute un feature supplementaire sur les vecteurs d'un jeu de donnees ####


def addBiasToDataset(X):
	if X.ndim == 1:
		np.reshape(X,(X.shape[0],1))
		
	return np.lib.pad(X, (1,0), 'constant', constant_values=(1))[1:]