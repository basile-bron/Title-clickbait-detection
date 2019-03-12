import numpy as np
import os
import math
import sys
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import gc
#custom
from data_import import videos,logger
################################################################################
#Vectorize function
def vectorize(titles,model, videos):
	#logger.debug("Vectorizing CSV : %s",len(titles)," titles")
	x = np.zeros((len(titles),110,300))
	y = np.zeros((len(titles)))
	clean_titles = np.zeros((len(titles),110))
	n=0
	print("result shape will be : ", x.shape)

	for video in videos:

		#debug
		print(video.ratings)

		titles[n] = str(titles[n])
		np.append(y, video.ratings)
		np.append(x[n][0], video.number_of_capital_letter)
		np.append(x[n][1], video.number_of_exclamation_point)
		np.append(x[n][2], video.number_of_interogation_point)

		np.append(clean_titles[n][0], video.number_of_capital_letter)
		np.append(clean_titles[n][1], video.number_of_exclamation_point)
		np.append(clean_titles[n][2], video.number_of_interogation_point)
		w=3

		for word in titles[n].split():

			#print('n :',n,' w :', w)
			#convert to str to clean quote
			word = str(word)
			word = word.replace("'", "")

			np.append(clean_titles[n][w], word)

			try :
				#vectorize
				np.append(x[n][w], model.word_vec(word))
				#print('word',word,'in voc')
				#print('vector is : ',word_vector)
				##print('of shape : ',word_vector.shape)
			except KeyError:

				#print('word not in voc')
				pass
			w = w+1
		n = n+1
	print(x.shape)
	print(x[10])


	return x, y, clean_titles
##########################################################
#loading the google pretrained vocabulary for vectorization (for later use in the vectorize function, it's a big file so we dont want to load it each time we vectorize)
print("loading google word2vec model (can take a few minutes)")
logger.debug("loading google word2vec model")
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)

titles = [video.title for video in videos]
X = []
X, Y, clean_titles = vectorize(titles, model, videos)

print(X.shape)

print('END OF VECTORIZATION')
print('##################################')
############################################################
