import cv2
import numpy as np
import control as ct
from DNF import DNF
from scipy import ndimage



# TODO : Pour installer opencv:
# sudo apt-get install opencv* python3-opencv

# Si vous avez des problèmes de performances
#
# self.kernel = np.zeros([width * 2, height * 2], dtype=float)
# for i in range(width * 2):
#     for j in range(height * 2):
#         d = np.sqrt(((i / (width * 2) - 0.5) ** 2 + ((j / (height * 2) - 0.5) ** 2))) / np.sqrt(0.5)
#         self.kernel[i, j] = self.difference_of_gaussian(d)
#
#
# Le tableau de poids latéreaux est calculé de la façon suivante :
# self.lateral = signal.fftconvolve(self.potentials, self.kernel, mode='same')

size = (96, 128)
global DNF

def selectByColor(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_blue = np.array([80,50,50])
	upper_blue = np.array([130,255,255])
	framee = cv2.inRange(hsv, lower_blue, upper_blue)
	resha = reduire(framee,size)
	DNF.input = resha
	DNF.update_map()
	return framee

def reduire(a , shape) :
	sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
	return a.reshape(sh).mean(-1).mean(1)
	
def findCenter():
    # TODO trouver les coordonnées du centre à suivre
	bx, by = ndimage.measurements.center_of_mass(DNF.potentials)
	
		
	return (bx,by)

def motorControl(x,y):
    # TODO utilisez la fonction ct.move pour déplacer la caméra
	sx = x-size[0]/2
	sy = y-size[1]/2
	#normalize
	sx = sx / size[0]
	sy = sy / size[1]
	# vitesse entre -3 et 3
	sx = sx *30
	sy = sy *30
	print("vitesse ", sx,sy)
	ct.move(sx,sy)
	pass

def track(frame):
	input = selectByColor(frame)
	a,b = findCenter()
	print("centre",a , b)
	res = cv2.bitwise_and(frame,frame,input)
	im = cv2.circle(res,(int(b),int(a)),10,(0,0,255),-1)
	im = cv2.circle(im,(int(size[1]/2),int(size[0]/2)),5,(0,255,0),-1)
	im = cv2.rectangle(im, (0,0), (size[1],size[0]), (255,0,0)) 
	cv2.imshow("centre", im)
	cv2.imshow("dnf", input)
	#cv2.imshow("DNF", DNF.potentials)
	#print(DNF.potentials)
# DNF.input = input
# DNF.update_map()
# cv2.imshow("Input", DNF.input)
# cv2.imshow("Potentials", DNF.potentials)
	#center = findCenter()
	motorControl(a,b)

if __name__ == '__main__':
	cv2.namedWindow("Camera")
	vc = cv2.VideoCapture(0) # 2 pour la caméra sur moteur, 0 pour tester sur la votre.

	if vc.isOpened(): # try to get the first frame
		rval, frame = vc.read()
		DNF = DNF(size[0],size[1])
		input = selectByColor(frame)
		# initiali0sez votre DNF ici
	else:
		rval = False

	while rval:
		#cv2.imshow("Camera", frame)
		rval, frame = vc.read()
		track(frame)
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break

	cv2.destroyAllWindows()
