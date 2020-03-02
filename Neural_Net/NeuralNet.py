import numpy as np
import h5py
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

EP = []

def entropy(a,b):
	I = np.identity(len(a))
	y = I[:,b]
	e = np.dot(y.T, np.log(a))
	return(e)

def softmax(a):
	m = np.exp(a)
	for i in range(len(a)):
		a[i] = math.exp(a[i])/np.sum(m)
	return(a)


def derivative(x):
	for i in range(len(x)):
		x[i] = x[i]*(1.0-x[i])
	return (x)

def sigmoid(x):
	# print(x)
	for i in range(len(x)):
		x[i] = 1.0/(1.0 + np.exp(-x[i]))
	return(x)

def activation(w, b, x):
	a = np.dot(w,x)
	return(a+b)

def forward_propagation(w, b, x):
	output = activation(w, b, x)
	output = sigmoid(output)
	return(output)

def backward_propagation(layer,nodes,weight,intercept,output,t):
	I = np.identity(nodes[-1])
	true = I[:,t]
	error = []
	D = []
	j = 0
	for i in range(layer-2,-1,-1):
		# print("IN")
		if(i!=layer-2):
			e = np.dot(weight[i+1].T,delta)
			error.append(e)
		else:
			error.append(true - output[i])

		delta = error[j] * derivative(output[i])
		# print(delta.shape)
		D.append(delta)
		j+=1

	return(error[::-1], D[::-1])


def update_weights(layer,nodes,weight,intercept,output,delta,points,alpha=0.1):
	# print(np.array(output[0]).shape)
	# print(weight[1].shape)
	points = points[:,0]
	for i in range(layer-1):
		# print(weight[0][0][0])
		# print(intercept[0][0])
		# print("IN")
		# for j in range(nodes[i+1]):
		# 	if(i==0):
		# 		weight[i][:,j] = weight[i][:,j] + (alpha * delta[i] * points[:,0])
		# 		# weight[i] += alpha * np.dot(points, delta[i])
		# 	elif(i==layer-2):
		# 		weight[i][j] = weight[i][j] + (alpha * delta[i] * output[i-1][j])
		# print(delta[i].shape)
		if(i==0):
			weight[i] += alpha * np.outer(delta[i],points.T)
		else:
			weight[i] += alpha * np.outer(delta[i], output[i-1].T)
		intercept[i] += alpha * delta[i]
		# print(weight[0][0][0])
		# print(intercept[0][0])
	return(weight, intercept)

def make_vector(a):
	x = np.matrix(a)
	# print(x)
	return (x.flatten().T)

def predict(points, labels, weights, intercept, layer):
	c = 0.0
	for i in range(len(labels)):
		output = np.array(make_vector(points[i]))
		output = output[:,0]
		# print(output.shape)
		# break
		out = []
		for j in range(layer-1):
			output = forward_propagation(weights[j],intercept[j], output)
			# print(output.shape)
			
			if(j == layer-2):
				output = softmax(output)
				# print(np.amin(output))
				# print(np.argmax(output), labels[i])
				if(np.argmax(output)==labels[i]):
					c = c + 1.0
	print(c/len(labels))

def nn(layer, nodes, points, labels): # Layer includes input & output layer
	weights = []
	intercept = []
	for i in range(layer-1):
		w = []
		b = []
		for j in range(nodes[i+1]):
			w.append(np.random.randn(nodes[i]))
			b.append(np.random.randn(1)[0])
		# w = np.array(w)
		# print(w.shape)
		# print(np.array(w).shape)
		# print(np.array(b).shape)
		weights.append(np.array(w))
		intercept.append(np.array(b))

	weights = np.array(weights)
	intercept = np.array(intercept)

	for epoch in range(20):
		sum_error = 0.0
		for i in range(len(points)): # len(points)
			# print(weights[0][0][0])
			output = np.array(make_vector(points[i]))
			output = output[:,0]
			# print(output.shape)
			# break
			out = []
			for j in range(layer-1):
				output = forward_propagation(weights[j],intercept[j], output)
				# print(output.shape)
				
				if(j == layer-2):
					# output = softmax(output)
					sum_error += entropy(output, labels[i])

				out.append(output)

			error, delta = backward_propagation(layer, nodes, weights, intercept, out, labels[i])
				# points = output
			# print(np.array(error[1]).shape)
			# print(np.array(delta).shape)
			# print("DOEN")

			# weights = update_weights(layer, nodes, weights, intercept, out, labels[i])
			W, B = update_weights(layer, nodes, weights,intercept, out, delta, np.array(make_vector(points[i])))
			# print(W[0][0][0])
			weights = W
			intercept = B
			# print("DONE")
		EP.append(sum_error/(-1*len(points)))
		print("Epoch " + str(epoch) + ": " + str(sum_error/(-1*len(points))))

	return(weights, intercept)


if __name__ == '__main__':

	data = h5py.File('./data/Q1/MNIST_Subset.h5', 'r')

	points = np.array(data['X'])
	labels = np.array(data['Y'])

	# print(labels[:50])
	X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=0.2)
	# print(points[0].shape)
	weights, intercept = nn(3,[784,100,10], X_train, y_train)
	np.save('model1_sigmoid_weight', weights)
	np.save('model1_sigmoid_intercept', intercept)

	weights = np.load('model1_sigmoid_weight.npy')
	intercept = np.load('model1_sigmoid_intercept.npy')

	predict(X_test, y_test, weights, intercept, 3)
	predict(X_train, y_train, weights, intercept, 3)
	# print(len(EP))

	plt.figure()
	plt.plot(EP, c="blue", label='1 Hidden Layer')

	print("\n")
	print("\n")
	EP = []

	weights, intercept = nn(5,[784,100,50,50,10], X_train, y_train)
	np.save('model2_sigmoid_weight', weights)
	np.save('model2_sigmoid_intercept', intercept)

	weights = np.load('model2_sigmoid_weight.npy')
	intercept = np.load('model2_sigmoid_intercept.npy')

	predict(X_test, y_test, weights, intercept, 5)
	predict(X_train, y_train, weights, intercept, 5)

	plt.plot(EP, c="red", label='3 Hidden Layers')
	plt.legend()
	plt.show()


