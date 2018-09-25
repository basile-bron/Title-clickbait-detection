import numpy as np
import os
import math
import sys
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import gc
#custom
from import_data import titles, test_titles#titles #scores  #test_scores

################################################################################
#vectorize function
def vectorize(titles,model):


	#initialise in tree step because i don't have a pc from the nasa, and at least it is easier to understand
	#print("initialisation of the input")
	#the size of the vector of each word will be 300
	word_vector = [0 for x in range(300)]
	#the size of the title is 110 (this is arbitrary, d'ont hesitate to change it)
	title = [word_vector for j in range(110) ]
	#word_vector * number of words in title * number of titles
	result = [title for j in range(len(titles)) ]

	n=0
	for title in titles:
		w=0
		##print('title n:',n,' is ', title)
		title = str(title)
		for word in title.split():

			#print('n :',n,' w :', w)
			#convert to str to clean quote
			word = str(word)
			word = word.replace("'", "")

			try :
				#vectorize
				result[n][w] += model.word_vec(word)
				##print('word',word,'in voc')
				##print('vector is : ',word_vector)
				##print('of shape : ',word_vector.shape)
			except KeyError:
				#print('word not in voc')
				pass
			w = w+1
		n = n+1


	return result
##########################################################
#the moiise function
def splitter(X):
	#loading the google pretrained vocabulary for vectorization (for later use in the vectorize function, it's a big file so we dont want to load it each time we vectorize)
	print("loading google word2vec model")
	model = KeyedVectors.load_word2vec_format('real_data/GoogleNews-vectors-negative300.bin',binary=True)

	max = len(X)
	print('max ', max)

	#batch size is the size of array you will use,
	batch_size = 5000
	number_of_batch = max/batch_size
	#we need to rounded up the nuber because we cant do a loop with a float obviusly
	#PRECISION : we need to round to the upper number because,exemple : if our dataset is 157 and we want to split
	#in batchs of 10 value then the last seven value also need to be used
	number_of_batch = math.ceil(number_of_batch)

	i=0
	X_vectorized = [None]*number_of_batch

	for i in range(number_of_batch):

		#####################BATCHING##############################
		print('batch number : ', i,'/',number_of_batch)
		print('####################################################')

		if i != (number_of_batch-1):

			#do the math, we need to start at 0 at the begining and at "batch_size" whe i =1 and etc... so :
			start = batch_size*i
			#same. do the math
			end = batch_size*(i+1)
			#print(i,' batch, start at :',start ,'and end at :',end)
			#schrinked X of size batch_size
			x_ = X[start:end]
			#exeption for the last batch
		elif i == number_of_batch-1:
			start = batch_size*i
			end = max
			#print(i,' batch, start at :',start ,'and end at :',end)
			x_ = X[start:end]
		else:
			#no exeption always add one
			print('wtf happend man !')


		########################VECTORIZE##########################
		#initialise the output for the full vectorized array
		#X_vectorized = [None]*number_of_batch
		X_vectorized[i] = vectorize(x_, model)
		i=i+1
	########################RESTORE##########################
	print('restoring the input')
	X = X_vectorized[0]
	for i in range(len(X_vectorized)-1):
		print(i,'iteration')
		X.extend(X_vectorized[i+1])
	#X.append(X_vectorized[0:len(X_vectorized)][:])
	#print the full vector in a file
	#f = open(filename,'w', encoding='utf-8')
	#np.set_printoptions(threshold=np.nan)
	#print(X, file=f)
	#

	return X

		#f = open(filename,'w', encoding='utf-8')
		#np.set_printoptions(threshold=np.nan)
		#print(data, file=f)

#########################################################
print("VECTORIZATION")

vectorized_test_titles = splitter(test_titles)
#save('data/vectorized_test.txt', vectorized_test_titles)

vectorized_train_titles = splitter(titles)
#save('data/vectorized_train.txt', vectorized_train_titles)


#Debuging info
#print('vectorized_train_titles shape :', len(vectorized_train_titles))
print('vectorized_test_titles shape :', len(vectorized_test_titles))
print('END OF VECTORIZATION')
print('##################################')
############################################################
