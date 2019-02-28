import numpy as np
import os
import math
import sys
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import gc
#custom
from data_import import videos

################################################################################
#Vectorize function
def vectorize(titles,model):

	#initialise in tree step because i don't have a pc from the nasa, and at least it is easier to understand
	#print("initialisation of the input")
	#the size of the vector of each word will be 300
	word_vector = [0 for x in range(300)]
	#the size of the title is 110 becaus on average most of the title are less than that (this is arbitrary, d'ont hesitate to change it)
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
#loading the google pretrained vocabulary for vectorization (for later use in the vectorize function, it's a big file so we dont want to load it each time we vectorize)
print("loading google word2vec model")
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)

videos.title vectorize(videos.title, model)
print(videos.title.shape)
#for video in videos:
#	video.title = vectorize(video.title, model)

#Debuging info
#print('vectorized_train_titles shape :', len(vectorized_train_titles))
#print('vectorized_test_titles shape :', len(vectorized_test_titles))
print('END OF VECTORIZATION')
print('##################################')
############################################################
