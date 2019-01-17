import numpy as np
import math
import os
import time
from termcolor import colored
import matplotlib.pyplot as plt

from computeLoss import computeLoss
from gradientDescent import gradientDescent
from plotCurve import plotCurve
from plotData import plotData
from utils import pause

from addBiasToDataset import addBiasToDataset


#Partie I Visualisation du jeu de donnees
print colored("Partie 1 : Chargement et visualisation du jeu de donnees :  \n",'red', attrs=['bold'])

#On charge le tableau numpy
print("Chargement du jeu de donnee ..")
dataset = np.load("data1.npy")
print("Affichage du dataset : ")
print dataset
print("\n")


#Recuperation du nombre d'exemples d'apprentissage ainsi que la dimension des vecteurs
n_samples = dataset.shape[0]
print("Nombre d'exemples d'apprentissage n_samples = %d " % n_samples)
print("\n")

#On separe les valeures d'entree et les valeures a predire dans deux tableau differents
print("On recupere la premiere colonne pour les donnees d'entree :")
X_plot = np.zeros((n_samples,1))
X_plot[:,0] 		= dataset[:,0]

##############Changez les la place des commentaires pour travailler en vectoriel
X = X_plot
# = addBiasToDataset(X_plot)
################################################################################

print("X = ")							 
print X.ndim					 

print("On recupere la deuxieme colonne pour les valeures a predire :")
y = dataset[:,1]
print("y = ")
print y
print("\n")

#Utilisation de matplotlib pour visualiser sur un graphique les donnees
plotData(X_plot, y)
pause()
#################################################################################

print colored("Partie 2 : Calcul de la fonction de cout : \n",'red', attrs=['bold'])

print("Initialisation aleatoire de notre vecteur de parametre et affichage sur le graphique :")
theta = np.zeros(2) #np.random.normal(0,1,2)
print("theta = ")
print theta

line_X = np.arange(X.min(), X.max())
line_y = theta[0] + line_X * theta[1]

plt.plot(line_X, line_y, color='navy', label='Random parameters')
plt.show(block=False)

# Completez les fichiers predictLinReg computeLoss avant de poursuivre
J = computeLoss(X, y, theta)
print("La valeur initiale de la loss est J = %f" % J)
pause()
plt.clf()
###################################################################################

print colored("Partie 3 : Calcul du gradient de la fonction de cout et mise a jour des parametres du modele : \n",'red', attrs=['bold'])
learning_rate = 0.01
n_iter = 1500

start_time = time.time()
print("Apprentissage en cours ...")
theta, costHistory = gradientDescent(X, y, theta, learning_rate, n_iter)
stop_time =  time.time()
print("\n")

print("\nTemps de calcul = %f secondes" % (stop_time - start_time))
print("\n")

print("Vecteur de parametres retourner par l'algorithme :")
print theta
print("\n")
pause()
###################################################################################

print colored("Partie 4 : Affichage de notre brave modele appris : \n",'red', attrs=['bold'])
plotData(X_plot, y)
line_X = np.arange(X.min(), X.max())
line_y = theta[0] + line_X * theta[1]


plt.plot(line_X, line_y, color='navy', label='Linear regressor')
plt.show(block=False)
pause()
plt.clf()
###################################################################################

print colored("Partie 5 : Affichage de la fonction de loss au cours du temps : \n",'red', attrs=['bold'])
plotCurve(costHistory)

pause()
plt.clf()
###################################################################################