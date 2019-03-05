import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import cv2

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import logistic


np.set_printoptions(threshold=np.inf)
class dnf:
	def __init__(self):
		self.iteration = 0 
		self.sig = 2
		self.taille = 30
		self.p1 = [self.taille*.25,self.taille*.25]
		self.p2 = [self.taille*.75,self.taille*.75]
		self.dnf = np.random.rand(self.taille,self.taille)
		self.lat = np.zeros([self.taille,self.taille], dtype=float)
		self.inputt = np.zeros([self.taille,self.taille], dtype=float)
		self.tay = self.taille*2
		self.kernel = np.zeros([self.tay, self.tay], dtype=float)

		self.tker2 = self.taille*3
		self.sigker2 = 2
		self.kernel2 = np.zeros([self.tker2, self.tker2], dtype=float)

		self.lateral = np.zeros([self.taille, self.taille], dtype=float)
		self.temps = 1 
		self.to = 0.64
		self.dt = 0.1
		self.clik = 10
		self.vitesse = self.clik+1
		self.ampli = 0.25
		self.vitesse = 0.25

		self.sens =  1

		fool = 0.5
		for i in range(self.tay):
			for j in range(self.tay):
				d = np.sqrt(((i/(self.tay)-fool) ** 2 + ((j/(self.tay)-fool) ** 2))) / np.sqrt(fool)
				self.kernel[i, j] = self.difference_of_gaussian(d)

		for i in range(self.tker2) :
			for j in range(self.tker2) :
				d = self.euclidien_dist((i,j),(self.tker2/2,self.tker2/2))
				self.kernel2[i,j] = self.difference_of_gaussian(d/self.tker2)

		k2m = np.max(self.kernel2)
		self.kernel2 /= k2m

	def euclidien_dist(self,x,y): # x , y vecteur de meme taille
		return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

	def func (self,val, sig , mu):  # mu = moyenne , sig = variance

		t = (val-mu)/sig

		t = t*t

		a = (1.0/(sig*math.sqrt(2*math.pi)))*math.exp(-(1.0/2.0)*t)

		return a

	def func2 (self,val, sig = None):  # mu = moyenne , sig = variance
		if sig is None :
			sig = self.sig
		r = ((val)/sig)**2;
		r /= 2
		r = -r
		r = np.exp(r)

		return r;

	def gaussian_activity(self,	a = None, b = None, sig = None):# a et b centre, sig 
		if a is None :
			a = self.p1
		if b is None :
			b = self.p2
		if sig is None :
			sig = self.sig

		#print(self.vitesse,self.clik)

		for x in range(self.taille) :
			for y in range(self.taille) :
				self.inputt[x][y] = self.func2(self.euclidien_dist(a,[x,y]))+self.func2(self.euclidien_dist(b,[x,y]))

		top = np.max(self.inputt)
		
		self.inputt /= top

	def gaussian_activity_bruit_G(self,	a = None, b = None, sig = None,ampli = None):# a et b centre, sig 
		if a is None :
			a = self.p1
		if b is None :
			b = self.p2
		if sig is None :
			sig = self.sig
		if ampli is None :
			ampli = 0.25

		#bruit

		bruit = np.zeros([2,10])
		for i in range(10) :
			bruit[0,i] = np.random.rand(1)*45
			bruit[1,i] = np.random.rand(1)*45
		
		#print(self.vitesse,self.clik)

		for x in range(self.taille) :
			for y in range(self.taille) :
				self.inputt[x][y] = self.func2(self.euclidien_dist(a,[x,y]))+self.func2(self.euclidien_dist(b,[x,y]))
				for i in range(10) :
					self.inputt[x][y] += self.func2(self.euclidien_dist([bruit[0,i],bruit[1,i]],[x,y]))


		top = np.max(self.inputt)
		
		self.inputt /= top

	def gaussian_activity_bruit(self,	a = None, b = None, sig = None,ampli = None):# a et b centre, sig 
		if a is None :
			a = self.p1
		if b is None :
			b = self.p2
		if sig is None :
			sig = self.sig
		if ampli is None :
			ampli = 0.1

		
		#print(self.vitesse,self.clik)

		for x in range(self.taille) :
			for y in range(self.taille) :
				self.inputt[x][y] = self.func2(self.euclidien_dist(a,[x,y]))+self.func2(self.euclidien_dist(b,[x,y]))
				#if self.clik >= self.vitesse :
				if np.random.rand(1) > 0.9 :
					#self.inputt[x][y] += self.func2(self.euclidien_dist((self.taille/2,self.taille/2),[x,y]),self.taille)
					self.inputt[x][y] += np.random.rand(1)*ampli
				else :
					self.inputt[x][y] -= np.random.rand(1)*ampli
					#self.inputt[x][y] += np.random.multivariate_normal(np.zeros(30),np.identity(30))


		top = np.max(self.inputt)
		
		self.inputt /= top
		 

	def difference_of_gaussian(self,distance):

		ce = 1.25 
		pe = 0.05
		ci = 0.7
		pi = 10 

		g = ce*np.exp(-(distance**2/(2*pe**2)))
		d = ci*np.exp(-(distance**2/(2*pi**2)))

		som = (g-d)

		return som

	def somm2(self,g,h) :
		for o in range(self.taille) :
				for p in range(self.taille) :
					if (o,p) != (g,h) :
						self.lat[g][h]  +=  self.dnf[o][p]*self.difference_of_gaussian(self.euclidien_dist([g,h],[o,p]))

		h = np.max(self.lat)


	def laterall(self) :

		tmp = np.zeros([self.taille,self.taille])
		for o in range(self.taille) :
			for p in range(self.taille) :
				for x in range(self.taille) :
					for y in range(self.taille) :
						if (o,p) != (x,y) :
							#tmp[o][p]  +=  self.dnf[o][p]*self.difference_of_gaussian(self.euclidien_dist([x,y],[o,p]))
							tmp[o][p]  +=  self.dnf[o][p]*self.func2(2)

		z = np.max(tmp)
		if z > 0 :
			tmp /= z

		self.lat = tmp
		
	def teral(self) :
		self.lateral = self.dnf#*self.kernel

	def move(self) :

		if self.p2[1] > self.taille*0.75 : 
			self.vitesse *= -1

		if self.p2[1] < self.taille*0.25 : 
			self.vitesse *= -1

		self.p2[1] += self.vitesse
		self.p1[1] -= self.vitesse

		self.p2[0] += self.vitesse
		self.p1[0] -= self.vitesse

	def cheat(self) :

		self.lateral = signal.fftconvolve(self.dnf, self.kernel2, mode='same')

		max = np.maximum(np.max(self.lateral), np.abs(np.min(self.lateral)))
		if max != 0 :
			self.lateral /= max
				

	def runge(self, inp, temps = None , dt = None) :

		if temps is None :
			temps = self.temps
		if dt is None :
			dt = self.dt

		rk1 = self.rk(inp,temps)
		rk2 = self.rk(inp+(dt/2.0)*rk1,temps+(dt/2.0))
		rk3 = self.rk(inp+(dt/2.0)*rk2,temps+(dt/2.0))
		rk4 = self.rk(inp+dt*rk3,temps+(dt))

		res = inp+(dt*(rk1+2.0*rk2+2.0*rk3+rk4))/6.0

		return res

	def rk(self,inp,t) :

		rk = 1.0 - (t * inp)

		return rk

	def update_neuron(self) :
		som = 0
		res = np.random.random((self.taille, self.taille))
		n = 0
		nn = np.inf
		self.gaussian_activity()
		self.cheat()

		for o in range(self.taille) :
			for p in range(self.taille) :
				self.dnf[o][p] += self.dt*((-self.dnf[o][p]+self.lateral[o][p]+self.inputt[o][p])/self.to)
		
				if self.dnf[o][p] < 0 :
					self.dnf[o][p] = 0
				if self.dnf[o][p]  > 1 :
					self.dnf[o][p] = 1

		self.temps += self.dt








	def update_neuron_mobil(self) :
		som = 0
		res = np.random.random((self.taille, self.taille))
		n = 0
		nn = np.inf
		self.move()
		self.gaussian_activity()
		self.cheat()

		for o in range(self.taille) :
			for p in range(self.taille) :
				self.dnf[o][p] += self.dt*((-self.dnf[o][p]+self.lateral[o][p]+self.inputt[o][p])/self.to)
		
				if self.dnf[o][p] < 0 :
					self.dnf[o][p] = 0
				if self.dnf[o][p]  > 1 :
					self.dnf[o][p] = 1

		self.temps += self.dt


	def update_neuron_bruit(self) :
		som = 0
		res = np.random.random((self.taille, self.taille))
		n = 0
		nn = np.inf
		self.gaussian_activity_bruit()
		self.cheat()

		for o in range(self.taille) :
			for p in range(self.taille) :
				self.dnf[o][p] += self.dt*((-self.dnf[o][p]+self.lateral[o][p]+self.inputt[o][p])/self.to)
				
				if self.dnf[o][p] < 0 :
					self.dnf[o][p] = 0
				if self.dnf[o][p]  > 1 :
					self.dnf[o][p] = 1

		self.temps += self.dt
		self.clik = self.clik +1

	def update_neuron_bruit_gauss(self) :
		som = 0
		res = np.random.random((self.taille, self.taille))
		n = 0
		nn = np.inf
		self.gaussian_activity_bruit_G()
		self.cheat()

		for o in range(self.taille) :
			for p in range(self.taille) :
				self.dnf[o][p] += self.dt*((-self.dnf[o][p]+self.lateral[o][p]+self.inputt[o][p])/self.to)
				
				if self.dnf[o][p] < 0 :
					self.dnf[o][p] = 0
				if self.dnf[o][p]  > 1 :
					self.dnf[o][p] = 1

		self.temps += self.dt
		self.clik = self.clik +1

	def update_neuron_bruit_vit(self, vit = None) :
		som = 0
		res = np.random.random((self.taille, self.taille))
		n = 0
		nn = np.inf
		if vit is not None :
			self.clik = vit

		if self.vitesse > self.clik :
			self.gaussian_activity_bruit()
			self.vitesse = 0
		self.cheat()

		for o in range(self.taille) :
			for p in range(self.taille) :
				self.dnf[o][p] += self.dt*((-self.dnf[o][p]+self.lateral[o][p]+self.inputt[o][p])/self.to)
				
				if self.dnf[o][p] < 0 :
					self.dnf[o][p] = 0
				if self.dnf[o][p]  > 1 :
					self.dnf[o][p] = 1

		self.vitesse +=1
		self.temps += self.dt

	def update_neuron_rk(self, dt = None) :

		if dt is not None :
			self.dt = dt

		som = 0
		res = np.random.random((self.taille, self.taille))
		n = 0
		nn = np.inf

		self.gaussian_activity()
		self.cheat()
				
		self.dnf =  self.runge(self.lateral) + self.inputt - self.dnf

		for o in range(self.taille) :
			for p in range(self.taille) :
				
				if self.dnf[o][p] < 0 :
					self.dnf[o][p] = 0
				if self.dnf[o][p]  > 1 :
					self.dnf[o][p] = 1


		self.temps += self.dt



def updatefig(*args):

	#bid.update_neuron()  # update neurone synchrone vanilla
	
	#bid.update_neuron_rk(0.01) # update neurone avec runge kutta probablement mal utilisé
								# le dnf et le latteral clignote mais parfois il arrive a ce stabiliser
								# argument = dt

	#bid.update_neuron_bruit() # update neurone synchrone vanilla avec bruit

	#bid.update_neuron_bruit_gauss() ## update neurone synchrone vanilla avec ajour bruit gaussien sur l inputt

	#bid.update_neuron_bruit_vit(5) # update neurone synchrone vanilla
									# avec vitesse du bruit gaussien reglable 
									# vitesse = valeur entiere
									# 0 est le plus rapide 

	#bid.update_neuron_mobil()

	im.set_array(bid.dnf)

	plt.subplot(2, 3, 2)
	im0 = plt.imshow(bid.inputt, cmap='hot', interpolation='nearest', animated=True)
	plt.title("inputt")
	plt.subplot(2, 3, 3)
	im1 = plt.imshow(bid.kernel, cmap='hot', interpolation='nearest', animated=True)
	plt.title("kernel")
	plt.subplot(2, 3, 4)
	im2 = plt.imshow(bid.lateral, cmap='hot', interpolation='nearest', animated=True)
	plt.title("lateral")
	plt.subplot(2, 3, 5)
	im3 = plt.imshow(bid.lat, cmap='hot', interpolation='nearest', animated=True)
	plt.title("lat")
	plt.subplot(2, 3, 6)
	im3 = plt.imshow(bid.kernel2, cmap='hot', interpolation='nearest', animated=True)
	plt.title("kernel2")

	return im,im2,im0


if __name__ == '__main__':
	bid = dnf()
	fig = plt.figure()
	bid.gaussian_activity()
	plt.subplot(2, 3, 1)
	im = plt.imshow(bid.dnf, cmap='hot', interpolation='nearest', animated=True)
	plt.title("dnf")
	ani = animation.FuncAnimation(fig, updatefig, interval=5, blit=True)
	plt.show()