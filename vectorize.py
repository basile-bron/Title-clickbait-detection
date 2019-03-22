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
	clean_titles = [["" for x in range(0,110)] for y in range(len(titles))]
	n=0
	print("result shape will be : ", x.shape)

	for video in videos:

		#debug

		titles[n] = str(titles[n])
		y[n] = video.ratings
		print(video.ratings)
		print(y)
		x[n][0] = video.number_of_capital_letter
		x[n][1] = video.number_of_exclamation_point
		x[n][2] = video.number_of_interogation_point

		clean_titles[n][0] = video.number_of_capital_letter
		clean_titles[n][1] = video.number_of_exclamation_point
		clean_titles[n][2] = video.number_of_interogation_point
		w=3

		for word in titles[n].split():

			#print('n :',n,' w :', w)
			#convert to str to clean quote
			word = str(word)
			word = word.replace("'", "")


			try :
				#vectorize
				""" pourquoi append et pas egale? et pk clean title me donne des chifffre mamene"""
				x[n][w] = model.word_vec(word)
				clean_titles[n][w] = word
				#print(n,w)
				#print(clean_titles[n][w])
				#print(x[n][w])
				#print('word',word,'in voc')
				#print('vector is : ',word_vector)
				##print('of shape : ',word_vector.shape)
			except KeyError:

				#print('word not in voc')
				pass
			w = w+1
		n = n+1


	return x, y, clean_titles
##########################################################
#loading the google pretrained vocabulary for vectorization (for later use in the vectorize function, it's a big file so we dont want to load it each time we vectorize)
print("loading google word2vec model (can take a few minutes)")
logger.debug("loading google word2vec model")
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)

titles = [video.title for video in videos]
X = []
X, Y, clean_titles = vectorize(titles, model, videos)
print(len(Y))
"""
with open('your_file.txt', 'w') as f:
    for item in clean_titles:
        f.write("%s\n" % item)
"""
print(X.shape)

print('END OF VECTORIZATION')
print('##################################')
############################################################
