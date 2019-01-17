import numpy as np
import math
import os
import time
from termcolor import colored
import matplotlib.pyplot as plt

from non_linearities import sigmoid


from computeLossRegLog import computeLoss
from gradientDescentRegLog import gradientDescent
from plotCurve import plotCurve
from plotData import plotData, plotData2D
from utils import pause

from addBiasToDataset import addBiasToDataset


#Partie I Visualisation du jeu de donnees
print colored("Partie 1 : Chargement et visualisation du jeu de donnees :  \n",'red', attrs=['bold'])

#On charge le tableau numpy
print("Chargement du jeu de donnee ..")
dataset = np.load("data2.npy")
print("Affichage du dataset : ")
print dataset
print("\n")


#Recuperation du nombre d'exemples d'apprentissage ainsi que la dimension des vecteurs
n_samples = dataset.shape[0]
print("Nombre d'exemples d'apprentissage n_samples = %d " % n_samples)
print("\n")

#On separe les valeures d'entree et les valeures a predire dans deux tableau differents
print("On recupere les deux dernieres colonnes pour les vecteur d'entree :")
X_plot = dataset[:,1:]
X = addBiasToDataset(X_plot)
print("X = ")
print X

print("On recupere la premiere colonne pour les labels :")
y =dataset[:,0]
print("y = ")
print y
print("\n")

#Utilisation de matplotlib pour visualiser sur un graphique les donnees
plotData2D(X_plot, y)
pause()
#################################################################################

print colored("Partie 2 : Calcul de la fonction de cout : \n",'red', attrs=['bold'])

print("Verification de la fonction sigmoid sur un vecteur")
print sigmoid(np.zeros(3))

print("Initialisation aleatoire de notre vecteur de parametre et affichage sur le graphique :")
theta = np.zeros(3)
print("theta = ")
print theta

line_X = np.arange(X.min(), X.max())
line_y = theta[0] - line_X * theta[1] / theta[2] 

plt.plot(line_X, line_y, color='navy', label='Random parameters')
plt.show(block=False)

# Completez les fichiers predictLogReg computeLossRegLog avant de poursuivre
J = computeLoss(X, y, theta)
print("La valeur initiale de la loss est J = %f" % J)
pause()
plt.clf()
###################################################################################

print colored("Partie 3 : Calcul du gradient de la fonction de cout et mise a jour des parametres du modele : \n",'red', attrs=['bold'])
learning_rate = 0.001
n_iter = 300

# Completez le fichier gradientDescentRegLog avant de poursuivre
print("Apprentissage en cours ...")
theta, costHistory = gradientDescent(X, y, theta, learning_rate, n_iter)
print("\n")

print("Vecteur de parametres retourne par l'algorithme :")
print theta
print("\n")

plotCurve(costHistory)
pause()
plt.clf()
###################################################################################

print colored("Partie 4 : Affichage de notre brave modele appris : \n",'red', attrs=['bold'])
plotData2D(X_plot, y)
line_X = np.arange(X.min(), X.max())
line_y = theta[0] - line_X * theta[1] / theta[2]


plt.plot(line_X, line_y, color='navy', label='Decision boundary')
plt.show(block=False)


pause()
plt.clf()
###################################################################################

print colored("C\'EST FINI !!!",'green',attrs=['bold'])