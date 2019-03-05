import math
import matplotlib.pyplot as plt
from random import gauss

print("dnf")



def euclidien_dist(x,y): # x , y vecteur de meme taille
	#print("machin")
	a = [0,0]
	
	for i in range(0,2) :
		#print(x[i],y[i])
		
		a[i] = x[i]-y[i]
		a[i] = a[i]*a[i]
		

	b = a[0]+a[1]	
	b = math.sqrt(b)
	return b
		
def gaussienne_activity(a , b , sig):# a et b centre, sig 

	grille = []
	for i in range(45):
		grille.append([i+0.0] * 45)
	#grille = []
	#grid = [[0] * 45 for _ in range(45)]
	for i in grille : 
		#print()
		for y in i :
			y = gauss(a,b)
			#print(y)

	x = [gauss(100,15) for i in range(10000)]

	num_bins = 50
	n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)

	plt.show()
	#print(grille[0][1])
def main() :

	x = [0,0]
	y = [1,0]
	
	d = euclidien_dist(x,y)
	print(d)
	
	gaussienne_activity(0,0,3)
	
main()