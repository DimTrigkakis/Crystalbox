from crystalbox import *
 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers import *
from keras.models import Model

import numpy as np
import random

class Node:
	
	def __init__(self):
		self.my_input = []
		self.my_output = None
	
	def dense_connect(self,input_node,connect=16):
		self.my_input.append(input_node.output())

		input_merge = None
		if len(self.my_input) > 1:
			print(self.my_input)
			input_merge = merge(self.my_input, mode="concat")
		else:
			input_merge = self.my_input[0]

		self.my_output = Dense(connect,activation="relu")(input_merge)
		
	def direct_dense_connect(self,source,connect=16):
		self.my_output = Dense(connect, activation="relu")(source)

	def calculate_connection_complexity(self):
		return 1

	def input(self):
		return self.my_input
	
	def output(self):
		return self.my_output
		
class NodeLayer:

	def __init__(self):
		self.Nodes = []

	def add_Nodes(self,n=1):
		for i in range(n):
			print str(self)+" "+str(i)
			N = Node()
			self.Nodes.append(N)

	def direct_connect(self,source):
		p = 0.1
		for n in self.Nodes:
			n.direct_dense_connect(source)
	
	def random_connect_layer(self,layer):
		print "HELLO"
		print "LAyer with node number"+str(len(self.Nodes))+" "+str(len(layer.Nodes))
		p = 0.01
		for n in layer.Nodes:
			for n2 in self.Nodes:
				if random.random() > p:
					n2.dense_connect(n)
class Network:

	def __init__(self,source,architecture=[1]):

		self.NLayers = []
		self.output = None

		self.add_layer(input_layer=model_input)
		flag = "first"
		for x in architecture:	
			if flag != "first":
				self.add_layer(n=x)
			else:
				flag = "second"

	def add_layer(self,n=1,input_layer=None):
		l = NodeLayer()
		print "already here nodes"+str(len(l.Nodes))
		print "adding nodes" +str(n)+" for layer"+str(l)
		l.add_Nodes(n=n)
		print "added layer"+str(len(l.Nodes))	
		if input_layer == None:
			l.random_connect_layer(self.NLayers[-1])
		else:
			l.direct_connect(input_layer)
		
		self.NLayers.append(l)
		self.output = l.Nodes[0].output()

	def final_output(self):
		return self.output

	def get_layer(self,i):
		if i > len(self.NLayers):
			return None
		return self.NLayers[i]

model_input = Input(shape=(784,), dtype='float32',name="model_input")
model_input_re = Reshape((1,28,28),input_shape=(784,)) (model_input) 
#graph_network = Network(source=model_input,architecture=[1])
#t = graph_network.final_output()
model_hidden = Convolution2D(32,3,3, border_mode='valid', input_shape=(1,28,28)) (model_input_re)
model_hidden_1 = Convolution2D(16,3,3, border_mode='valid') (model_hidden)

model_hidden_2 = Flatten() (model_hidden_1)
model_out = Dense(64,activation="relu") (model_hidden_2)
model_output = Dense(10, activation="sigmoid", name="main_output")(model_out)

model = Model(input=[model_input], output=[model_output])
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),metrics=['accuracy'])

train_images, train_labels = Core.load_mnist(path="./MNIST_data/")

Draw.clear()

# Draw the first image

Draw.draw_symbol_image(train_images[random.randint(1,100)]/255.0,warp_size=[32,32])

train_images = train_images.reshape((train_images.shape[0],train_images.shape[1]*train_images.shape[2]))/255.0
train_labels_one_hot = np.zeros((train_labels.shape[0],10))
train_labels_one_hot[np.arange(train_labels.shape[0]), train_labels.reshape(train_labels.shape[0])] = 1

model.fit(train_images, train_labels_one_hot,nb_epoch=10, batch_size=256)

test_images, test_labels = Core.load_mnist(path="./MNIST_data/",dataset="testing")

test_images = test_images.reshape((test_images.shape[0],test_images.shape[1]*test_images.shape[2]))/255.0
test_labels_one_hot = np.zeros((test_labels.shape[0],10))
test_labels_one_hot[np.arange(test_labels.shape[0]), test_labels.reshape(test_labels.shape[0])] = 1

evaluation = model.evaluate(test_images, test_labels_one_hot, batch_size=256)
Core.show(evaluation)

