import numpy as np
import matplotlib.pyplot as plt


##### 	Fonction permettant de tracer une courbe de valeures ####
#		
#		

def plotCurve(vals):
	t = np.arange(0, vals.shape[0], 10)


	plt.plot(t, vals[t])

	plt.ylabel("J")
	plt.xlabel("iteration")

	plt.show(block=False)