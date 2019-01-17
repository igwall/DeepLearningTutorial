import numpy as np
import matplotlib.pyplot as plt



##### 	Fonction permettant d'afficher un jeu de donnee ####
#		
#		
def plotData(X,y):
	plt.scatter(X, y, c="g", alpha=0.8, marker='o')
	plt.ylabel("y")
	plt.xlabel("x")


	plt.show(block=False)

def plotData2D(X,y):
	plt.scatter(X[:,0],X[:,1], c=y, alpha=0.8, marker='o')
	plt.ylabel("x_0")
	plt.xlabel("x_1")
	plt.axis([-15, 15, -15, 15])


	plt.show(block=False)