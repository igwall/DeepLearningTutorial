import numpy as np
import math
import os
import time
from termcolor import colored

from non_linearities import sigmoid, tanH, ReLU

from utils import pause
os.system('clear')
#Partie 0 : Rappels sur les operation sur les variables et structures de base 
print colored("Partie 0 : Rappels sur les operations sur les variables et structures de base \n",'red', attrs=['bold'])


#Exemples avec des booleens
print colored("Exemples d'operation sur les nombres",'blue', attrs=['bold'])

#Declarer un entier
x = 3
print("x = %d" % x)
print("Le type de x est type(x): %s" % type(x)) # Affiche "<class 'int'>"

#declarer un floatant
y = 4.2
print("y = %d" % y)
print("Le type de y est type(y): %s" % type(y)) # Affiche "<class 'float'>"

print()

#Exemple d'addition
print("x + 1 = %d" % (x+1))

#Exemple d'addition entre int et float (le resultat est au format float)
print("x + y = %f" % (x+y))     		#Affiche "x + y = 7.200000"

#Exemple d'incrementation
print("On Incremente x=%d de 2 en affectant automatiquement cette nouvelle valeur a x en faisant x += 2:" % x)
x += 2
print("La nouvelle valeur de x = %d" % x) 

#Autre exemples
print("x - 1 = %d" % (x - 1))   				#Affiche "x - 1 = 2"
print("x * 2 = %d" % (x * 2))  				#Affiche "x * 2 = 6"
print("x a la puissance 2: x **2 = %d" % (x **2))  	#Affiche "x a la puissance 2 = 9"

print("\n")

pause()
os.system('clear')
########################################################################################


#Exemples avec des booleens
print colored("Exemples avec des booleens",'blue', attrs=['bold'])
t = True
f = False

print(type(f)) # Affiche "<class 'bool'>"
print(t and f) # ET logique; Affiche "False"
print(t or f)  # OU logique ; Affiche "True"
print(not t)   # NON logique ; Affiche "False"

#Une comparaison (operateur == ) retourne un booleen
x=3; y=2
print(x == y) #Affiche "False"
print(x==3)   #Affiche "True"

print("\n")

pause()
os.system('clear')
##########################################################################

#Rappels sur les listes
print colored("Exemples d'operations sur les listes",'blue', attrs=['bold'])

#Declarer une liste vide
l_empty = []
print("Voici une list vide: l_empty = []")
print(l_empty)
print("\n")

#Declarer une liste contenant des nombres
print("Voici une list non vide: l = [11, 5, 9, 10]")
l = [11, 5, 9, 10]
print(l)
print("\n")

#Acceder au i-eme element d'une liste
print("Le premier element de la liste est l[0] = %d" % l[0])
print("\n")

#Affichage du nombre d'element dans la liste
print("Le nombre d'elements dans l est len(l): %d" % len(l))
print("Le nombre d'elements dans l_empty est len(l_empty): %d" % len(l_empty))
print("Le dernier element de la liste est l[len(l) - 1] = %d" % l[len(l) - 1])
print("\n")

#Parcourir les elements d'une liste


print("On boucle sur les elements de l pour les afficher un a un:")
for i in range(len(l)):
	print(l[i])
print("\n")

print("On boucle sur les elements de l pour les afficher un a un avec leur indice correspondant avec la primitive 'enumerate':")
for i, e in enumerate(l):
	print("Le %d eme element de la liste est l[%d] = %d" % (i, i, l[i]) )

pause()

os.system('clear')

######################################%d#######################################################


# Partie II : Manipulation des matrices et vecteurs : Numpy
print colored("Partie II : Manipulation des matrices et vecteurs : Numpy", 'red', attrs=['bold'])

print("\n")


print colored("Les vecteurs",'blue', attrs=['bold'])
#### ICI rien a faire. On montre simplement quelques exemples
#d'instantiations de parcours et d'acces aux valeurs des vecteurs

print("Instantiation d'un vecteur R 2 (2 dimension reelles): a = np.ndarray(2):") 
a = np.ndarray(2)  #Contient des valeurs tres proche de 0
print(a)  
print("\n")  

print("Vecteur nul dans R 2 (2 dimension reelles): b = np.zeros(2):")  #  # Affiche "array([ 0.,  0.])""
b = np.zeros(2)
print(b)  
print("\n")                    						

print("Vecteur remplie de 1 dans R 5: c = np.ones(5):")  #    # Affiche "array([ 1.,  1.,  1.,  1.,  1.])""
c = np.ones(5)  		
print(c)   
print("\n") 

print("Vecteur aleatoire dans R 4: d = np.random.random(4):")  #     
d = np.random.random(4)  	
print(d) 

print("\n")
pause()
os.system('clear')
##########################################################################################


print colored("Les operations sur les vecteurs",'blue', attrs=['bold'])
#Soit deux vecteurs aleatoires x et y dans R 10
print("Soit deux vecteurs aleatoires x et y dans R 10")
print("x = np.random.random(10)")
print("y = np.random.random(10)")
x = np.random.random(10)
y = np.random.random(10)

print("x=")
print(x)
print("\n")

print("y=")
print(y)
print("\n")

#Donnez la valeure de la 7eme composante de x
# Rappel: l'indexation commence a 0. Donc on parle de l'indice 6 ici ^^.
print("La valeur de la 7 eme composante est: x[6] = %f" % x[6])
print("\n")

#Recuperez les 3 premieres composantes du vecteur x
print("Recuperez les 3 premieres composantes du vecteur x: x[:3]")
print(x[:3])
print("\n")

#Recuperez la dimension du vecteur avec la primitive shape
print("Recuperez la dimension du vecteur avec la primitive shape: x.shape[0]")
dim = x.shape[0]
print("\n")

#Recuperez les 3 derniere composante composantes 
print("Recuperez les 3 dernieres composantes: x[dim-3:]")
dim = x.shape[0]
print( x[dim-3:] )
print("\n")

#Recuperez les composantes de 3 a la derniere
print("Recuperez les composantes de 3 a la derniere: x[3:]")
print(x[3:])
print("\n")

#Recuperez les composantes alant de l'indice 3 a 7
print("Recuperez les composantes alant de l'indice 3 a 7: x[3:7]")
print(x[3:7])
print("\n")

pause()
os.system('clear')
#########################################################

#Parcours des elements des vecteurs
print colored("Parcours des elements des vecteurs: systeme des indices de la structure array de Numpy",'blue', attrs=['bold'])

print("x=")
print(x)
print("\n")

print("y=")
print(y)
print("\n")

#Parcourez les valeur du vecteur y avec en bouclant sur les indices
print("Parcourez les valeur du vecteur x avec en bouclant sur les indices")
print("Dans un premier temps, on recupere la dimension du vecteur: dim_x = x.shape[0]")
dim_x = x.shape[0]
print("Puis, on parcour la structure avec un boucle for:")
for i in range(dim_x):
	print("La valeur de la %d eme composante est: %f" % (i,x[i]))
print("\n")

#Parcourez les valeurs du vecteur y avec en bouclant directement sur les elements du vecteur
#Note ici on n'incremente plus directement sur les indices mais sur les element du vecteur, 
#pour retrouver l'indice en cours, on doit faire l'incrementration a la main.
print("Parcourez les valeurs du vecteur x avec en bouclant directement sur les elements du vecteur")
i=0
for x_i in x:
	print("La valeur de la %d eme composante est: %f" % (i,x_i))
	i += 1
print("\n")

#Faites la meme chose avec la fonction enumerate pour avoir acces aux indices sans faire d'incrementation
print("Faites la meme chose avec la fonction enumerate pour avoir acces aux indices sans faire d'incrementation")
for i, x_i in enumerate(x):
	print("La valeur de la %d eme composante est: %f" % (i,x_i))
print("\n")

pause()
os.system('clear')
#######################################################################

print colored("PARTIE QUESTIONS SUR LES VECTEURS!!!",'green', attrs=['bold'])
#Implementez les operations manquantes ci dessous
print colored("Implementez les operations manquantes ci dessous",'blue', attrs=['bold'])
print("1) Addition des vecteur: x + y:")
print(x+y)
print("\n")

print("2) Combinaison lineaire des vecteurs 4.2 * x + 1.2 * y:")
print ("ICI TU DOIS CODER")
print("\n")

print("3) Lister avec une boucle les elements de x a la puissance 2:")
print ("ICI TU DOIS CODER")
print("\n")

print("4) Generer et mettre dans un vecteur elements de x a la puissance 2 en une seule commande:")
z = x ** 2
print(z)
print("\n")

print("5) La somme des elements de x avec une boucle:")
somme = 0.0
print ("ICI TU DOIS CODER")
print("\n")

print("6) La somme des elements de x avec la primitive 'sum' de numpy:")
somme = 0.0
print ("ICI TU DOIS CODER")
print("\n")

print("7) Calculer la norme L_2 de x avec une boucle en utilisant les points 3 et 5:")
norme = 0.0    #ICI TU DOIS CODER
print ("ICI TU DOIS CODER")
print("\n")


print("9) Calculer la norme L_2 de x sans boucle en utilisant les points 4 et 6:")
print("Hint: En une seule ligne de commande, tu devra renvoyer la norme")
norme = 0.0
print ("ICI TU DOIS CODER")
print("\n")

print("10) Calculer le produit scalaire entre x et y avec une boucle:")
dot_product = 0.0
print ("ICI TU DOIS CODER")
print("\n")

print("11) Calculer le produit scalaire entre x et y avec la primitive 'dot' de numpy:")
dot_product = np.dot(x,y)
print(dot_product)
print("\n")

pause()

os.system('clear')
############################################################

print colored("Les operations sur les matrices",'blue', attrs=['bold'])

print("Matrice nulle dans 2x2: M1 = np.zeros((2,2)):")  #          
M1 = np.zeros((2,2))
print(M1)   
print("\n")                   						

print("Matrice 3x3 remplie de 1: M2 = np.ones((3,3))  :")  #  
M2 = np.ones((3,3))  		
print(M2)  
print("\n")  

print("Matrice 4x4 aleatoire: M3 = np.random.random((4,4))	:")  #     
M3 = np.random.random((4,4))	
print(M3) 
print("\n")

print("Matrice identite 5x5: M4 = np.eye(5,5) 	:")  #     
M4 = np.eye(5,5) 		
print(M4) 
print("\n")

pause()
os.system('clear')
##################################################################


print colored("Trouver les dimensions du Numpy array",'blue', attrs=['bold'])
#Soit deux matrices 4x4 M1 et M2
#Ici on montre une autre maniere d'instancier une matrice en l'intialisant avec
# des valeures.
print("Soit deux matrices 4x4 M1 et M2")
print("On peut initialiser les matrices directement:")
print("M1 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [9,10,11,8]])")
print("M2 = np.array([[9,10,11,3], [5,6,7,8], [1,2,3,8], [9,10,11,7]])")

M1 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [9,10,11,8]])
M2 = np.array([[9,10,11,3], [5,6,7,8], [1,2,3,8], [9,10,11,7]])

print("M1=")
print(M1)
print("\n")

print("M2=")
print(M2)
print("\n")
###########################################################

#Donnez les dimensions de la matrice avec 'shape de numpy'
shape = M1.shape                
print("Les dimensions de la matrice M1: shape = M1.shape ")
print(shape)
print("\n")

#Donnez le nombre de lignes et le nombre de colonnes de M1 a partir du resultat precedent
print("nb_lignes de M1 shape[0] = %d" % shape[0])
print("nb_colonnes de M1 shape[1]= %d" % shape[1])
print("\n")
pause()
os.system('clear')
######################################################""


print colored("Parcourir les elements d'une matrice",'blue', attrs=['bold'])

print("Donnez la valeure de l'element de la 1er ligne et 3 eme colone de M1")
#Donnez la valeure de l'element de la 1er ligne et 3 eme colone de M1
# Rappel: l'indexation commence a 0.
print("Donnez la valeure de l'element de la 1er ligne et 3 eme colone de M1: M1_02 = M1[0,2]	")
M1_02 = M1[0,2]	
print("La valeur de la M1_12 eme composante est:  = %f" % M1_02)
print("\n")

#Parcourez les valeur de la matrice M2 avec en bouclant sur les indices
print("Parcourez les valeurs de la matrice M1 avec en bouclant sur les indices")
for i in range(shape[0]):
	for j in range(shape[1]):
		print("La valeur de la composante M1_%d%d est:  = %d" % (i, j, M1[i,j]))
print("\n")

#Parcourez les vecteurs lignes de la matrice M2 en bouclant directement sur les elements du vecteur
print("Parcourez les vecteurs lignes de la matrice M2 en bouclant directement sur les element de la structure:")
print colored("NOTE: Demandez a l'encadrant ici, si vous ne comprenez pas bien cette ligne.",'green', attrs=['bold'])
print("M2=")
print(M2)
print("\n")
for v_i in M2:
	print(v_i)
print("\n")

#Parcourez les vecteurs colonnes de la matrice M2 en bouclant directement sur les elements du vecteur
# Astuce, baladez vous dans les vecteurs lignes de la transpose de la matrice
print("Parcourez les vecteurs colonnes de la matrice M2 en bouclant directement sur les elements du vecteur")
print colored("NOTE: Demandez a l'encadrant ici, si vous ne comprenez pas bien cette ligne.",'green', attrs=['bold'])

for v_j in np.transpose(M2):
	print(v_j)
print("\n")

#Recuperez les 2 premier vecteurs ligne de M1 en une seule ligne de commande
#Quand on ne specifie pas les indices des 2 champs, on a implcitement access aux indices des lignes seuls
print("Recuperez les 2 1er vecteurs ligne de M1 en une seule ligne de commande")
print("M1=")
print(M1)
print("\n")

vecteur_col_first_2 = M1[:2]      #Equivalent a M1[:2 , :]
print(vecteur_col_first_2)
print("\n")

#Recuperer en une seule ligne de commande les 2 derniers vecteurs colones de la matrices M1
#Ici on veut agir sur les colonnes, on doit donc specifer les intervals sur les lignes ET les colones.
print("Recuperez en une seule ligne de commande les 2 derniers vecteurs colones de la matrice M1")
nb_col = M1.shape[1]
M1_transpose = np.transpose(M1)
vecteur_row_last_2 = M1_transpose[nb_col - 2:]
print(vecteur_row_last_2)
print("\n")

pause()

os.system('clear')

#Implementez les operations manquantes ci dessous
os.system('clear')
print colored("PARTIE QUESTIONS SUR LES MATRICES!!!",'green', attrs=['bold'])
print colored("Implementez les operations manquantes ci dessous",'blue',attrs=['bold'])
print("\n")

print("1) Addition de M1  M2:")
print ("ICI TU DOIS CODER")
print("\n")

print("2) Multiplication terme a terme de M1  M2:")
print ("ICI TU DOIS CODER")
print("\n")

print("3) Afficher avec une boucle les elements de M1 a la puissance 2:")
print ("ICI TU DOIS CODER")
print("\n")

print("4) Generer la matrice M3 contenant les elements de M1 a la puissance 2 en une seule commande:")
print("ICI TU DOIS CODER")
print("\n")

print("5) La somme des elements de M1 avec une boucle:")
somme = 0.0
print("ICI TU DOIS CODER")
print("\n")

print("6) La somme des elements de M1 avec la primitive 'sum' de numpy:")
somme = 0.0
print("ICI TU DOIS CODER")
print("\n")

print("7) Cherchez dans vos souvenir ou sur internet et implementez la formule permettant de calculer le produit matricielle (different du produit terme a terme !) entre M1 et M2 avec une boucle:")
#On va calculer le temps que cela met avec une boucle
#On comparera par la suite ce temps avec celui prit par une autre methode
start_time = time.time()
M1M2 = np.zeros(M1.shape)       
print("Dimensions du resultat du produit matriciel (ici les 2 matrices sont de meme dimension, donc le resultat aura egalement la meme dimension) :")
print(M1M2.shape)
print("ICI TU DOIS CODER")
stop_time =  time.time()

print("M1M2 = ")
print(M1M2)
print("\nTemps de calcul = %f secondes" % (stop_time - start_time))
print("\n")

print("8) Calculer le produit matricielle entre M1 et M2 avec la primitive 'dot' de numpy:")
start_time = time.time()
print("ICI TU DOIS CODER")
M1M2 = 0       #retourner le resultat
print(M1M2)
stop_time =  time.time()

print("M1M2 = ")
print(M1M2)
print("\nTemps de calcul = %f secondes" % (stop_time - start_time))
print("\n")

print("9) Soit la matrice M3 suivante (aleatoire mais ce n'est pas ce qui est important ici). Calculer le produit matricielle entre M1 et M3 avec la primitive 'dot' de numpy (attention aux dimensions des deux matrices !) :")
matrix_product = 0.0
M3 = np.random.random((10,4))        
print("M3  =")
print(M3)
print("\n")

print("M1M3  =")
M1M3 = 0
print("ICI TU DOIS CODER")
print(M1M3)
print("\n")

print("11) Calculer le produit matricielle entre <x , M3> avec la primitive 'dot' de numpy:")
print("shape of vector x:")
print(x.shape)
print("shape of Matrix M3:")
print(M3.shape)
xM1 = 0
print("ICI TU DOIS CODER")
print (xM1)
print("\n")

print("12) Calculer le produit matricielle entre <M3 , x> avec la primitive 'dot' de numpy (qu'est ce qui change ?) :")
print("HINT: Pensez a la transpose de vos structures pour aligner leurs dimensions !")
M1x = 0
print("ICI TU DOIS CODER")
print (M1x)
print("\n")
######################################################################

pause()

print colored("C\'EST FINI !!!",'green',attrs=['bold'])