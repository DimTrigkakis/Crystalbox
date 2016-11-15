import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import random

class Colors():
	Random, Same, White = range(3)

class Draw:

	CSI = "\x1B["
	reset = CSI+"m"
	@staticmethod
	def drawc(text, color=(1,1,1), color_offset = 40):
		
		r,g,b = tuple(int(x*(255-color_offset)+color_offset) for x in color)
		print Draw.CSI+"38;2;"+str(r)+";"+str(g)+";"+str(b)+"m" + text + Draw.reset,
		
		return

	@staticmethod
	def warp_numpy_image(numpy_image,size=[28,28]):
			
		original_size = numpy_image.shape
		
		target_size = size
		

		numpy_image_warped = np.zeros(target_size,dtype=numpy_image.dtype)
	
		print original_size
		color = 0
		if original_size[2] != None:
			color = 1 #print "COLOR"	
		
		target_size = size
		if color == 1:
			target_size = np.array([size[0],size[1],3]) 

		numpy_image_warped = np.zeros(target_size, dtype=numpy_image.dtype) 
		if color == 0:
			for w in range(target_size[0]):
				for h in range(target_size[1]):
						w_t, h_t = w*original_size[0]/(target_size[0]), h*original_size[1]/(target_size[1])
						numpy_image_warped[w,h] = numpy_image[w_t,h_t]
		else:
			for c in range(3):
	                        for w in range(target_size[0]):
	                                for h in range(target_size[1]):
        	                                        w_t, h_t = w*original_size[0]/(target_size[0]), h*original_size[1]/(target_size[1])
                	                                numpy_image_warped[w,h,c] = numpy_image[w_t,h_t,c]

		
		return numpy_image_warped

	@staticmethod
	def draw_symbol_image(numpy_image,characters = range(9789,9798),colors=Colors.Same, warp_size=[28,28], normalize=255.0):
		
		size = numpy_image.shape
		k = 2
		for i in range(3):
			if size[i] <= 4:
				k = i
				break

		print k
		print "The axis of color"
		if (k == 0):
			numpy_image = np.swapaxes(numpy_image,0,2)
			numpy_image = np.swapaxes(numpy_image,0,1)
		
		numpy_image = Draw.warp_numpy_image(numpy_image,size=warp_size)
		print numpy_image
		W = numpy_image.shape[0]
                H = numpy_image.shape[1]
                
                for w in range(W):
                        for h in range(H):
				if len(numpy_image.shape) == 2:
					c_t = numpy_image[w,h]
					c = c_t,c_t,c_t
				else:
					c = numpy_image[w,h]

		
				r,g,b = tuple(min(1,max(0,ccc+random.random()*0.3)) for ccc in c)
				
				if colors == Colors.Random:
					draw_color = [r/normalize,g/normalize,b/normalize]
				elif colors == Colors.Same:
					draw_color = c/normalize
				else:
					draw_color = c[0]/normalize,c[0]/normalize,c[0]/normalize					

				character = unichr(random.choice(characters))
                                Draw.drawc(character,color=draw_color)
                        print ""
                return


	@staticmethod
	def clear():
		print(chr(27)+"[2J")
		return

class Core:

	@staticmethod
	def show(o):
		print(o)
		return 

	@staticmethod
	def load_mnist(dataset="training", digits=np.arange(10), path="."):
		"""
		Loads MNIST files into 3D numpy arrays

		Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
		"""

		if dataset == "training":
			fname_img = os.path.join(path, 'train-images-idx3-ubyte')
			fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
		elif dataset == "testing":
			fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
			fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
		else:
			raise ValueError("dataset must be 'testing' or 'training'")

		flbl = open(fname_lbl, 'rb')
		magic_nr, size = struct.unpack(">II", flbl.read(8))
		lbl = pyarray("b", flbl.read())
		flbl.close()

		fimg = open(fname_img, 'rb')
		magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = pyarray("B", fimg.read())
		fimg.close()
		ind = [ k for k in range(size) if lbl[k] in digits ]
		N = len(ind)

		images = zeros((N, rows, cols), dtype=uint8)
		labels = zeros((N, 1), dtype=int8)	    

		for i in range(len(ind)):
			images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
			labels[i] = lbl[ind[i]]

		return images, labels
