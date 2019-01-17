import numpy as np
import math
import os
import time
from termcolor import colored
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from non_linearities import sigmoid


from computeLossRegLog import computeLoss
from gradientDescentRegLog import gradientDescent
from plotCurve import plotCurve
from plotData import plotData, plotData2D
from predictLogReg import predictLogReg as f
from utils import pause

from addBiasToDataset import addBiasToDataset



#Partie I Visualisation du jeu de donnees
print colored("Partie 1 : Chargement et visualisation du jeu de donnees :  \n",'red', attrs=['bold'])

#On charge le tableau numpy
print("Chargement du jeu de donnee ..")
kwargs = {}
batch_size=60000
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=60000, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=60000, shuffle=True, **kwargs)
print("Affichage du dataset : ")
X = []
t = []
for batch_idx, (data, target) in enumerate(train_loader):
	X = data.cpu().numpy()
	t = target.cpu().numpy()
print("\n")
X = np.reshape(X,(60000,28*28))
print X.shape
print t.shape

#Recuperation du nombre d'exemples d'apprentissage ainsi que la dimension des vecteurs
n_samples = X.shape[0]
print("Nombre d'exemples d'apprentissage n_samples = %d " % n_samples)
print("\n")

plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
	plt.subplot(10,20,i+1)
	plt.imshow(X[i,:].reshape([28,28]), cmap='gray')
	plt.axis('off')
plt.show()


n_classes = np.max(t) + 1
print("Nombre de classes d'objets = %d " % n_classes)
print("\n")

print("On transforme les labels en vecteur :")
y = np.zeros((60000, n_classes))
for i,l in enumerate(t):
	y[i][l] = 1
print y
print("\n")

X_test = []
t_test = []
for batch_idx, (data, target) in enumerate(test_loader):
	X_test = data.cpu().numpy()
	t_test = target.cpu().numpy()
print("\n")
X_test = np.reshape(X_test,(10000,28*28))
print X_test.shape

#Recuperation du nombre d'exemples d'apprentissage ainsi que la dimension des vecteurs
n_samples = X_test.shape[0]
print("Nombre d'exemples de test n_samples = %d " % n_samples)
print("\n")


#On separe les valeures d'entree et les valeures a predire dans deux tableau differents
print("On recupere les deux dernieres colonnes pour les vecteur d'entree :")
X = addBiasToDataset(X)
X_test = addBiasToDataset(X_test)


pause()
#################################################################################

print colored("Partie 2 : Calcul de la fonction de cout : \n",'red', attrs=['bold'])

print("Initialisation aleatoire de notre vecteur de parametre et affichage sur le graphique :")
theta = np.zeros((28*28 + 1, 10))
print("theta = ")
print theta


# Completez les fichiers predictLogReg computeLossRegLog avant de poursuivre
J = computeLoss(X, y, theta)
print("La valeur initiale de la loss est J = %f" % J)
pause()
plt.clf()
###################################################################################

print colored("Partie 3 : Calcul du gradient de la fonction de cout et mise a jour des parametres du modele : \n",'red', attrs=['bold'])
learning_rate = 0.1
n_iter = 10

# Completez le fichier gradientDescentRegLog avant de poursuivre
print("Apprentissage en cours ...")
theta, costHistory = gradientDescent(X, y, theta, learning_rate, n_iter, batch_size)
print("\n")

print("Vecteur de parametres retourne par l'algorithme :")
print theta
print("\n")

plotCurve(costHistory)
pause()
plt.clf()
###################################################################################


print colored("Partie 4 : Resultats de prediction : \n",'red', attrs=['bold'])

res =  np.mean(np.argmax(f(X_test,theta),axis=1) == t_test)
print("Precision de l'algorithme sur les donnees de test = %f" % res)

pause()
plt.clf()
###################################################################################

print colored("C\'EST FINI !!!",'green',attrs=['bold'])