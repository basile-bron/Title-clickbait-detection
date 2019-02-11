import numpy as np
import os
import sys
import re
################################################################################
#Import data function
def import_data(data_name, filename):
    data=[]
    with open(filename, encoding='utf-8') as inputfile:
        for line in inputfile:
            data.append(line.strip())
        data = np.array(data)
    print("import ",data_name," of size", data.shape)
    return data


#importing X
titles=[]
titles = import_data('titles', "real_data/train/titles_uniq.txt")

#importing Y
scores=[]
scores = import_data('scores',"real_data/train/score_uniq.txt")

#importing X_test
test_titles=[]
test_titles = import_data('test_titles',"real_data/test/titles_uniq.txt")

#importing Y_test
test_scores=[]
test_scores = import_data('test_scores', "real_data/test/score_uniq.txt")

print("END OF IMPORT")
print("#############################")
############################################################
def clean(titles):
    import keras

    from keras.preprocessing.text import text_to_word_sequence
    # define the document
    clean = [["" for x in range(100)] for y in range(len(titles))]
    i=0
    for title in titles:
        # tokenize the document
        #title = re.compile()
        result = text_to_word_sequence(title, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', split=' ')
        j=0
        for word in result:
            #print(i,j)
            #print(word)
            clean[i][j] += word
            j = j +1
        #print(clean[i])
        i = i +1
    clean = np.matrix(clean)
    return clean

#titles = clean(titles)
#print(titles)
print(titles.shape)
