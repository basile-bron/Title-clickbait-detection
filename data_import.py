import numpy as np
import os
import sys
import re
import json
import pandas as pd
from pandas.io.json import json_normalize
import csv
#from string import ascii_lowercase
################################################################################
#Import data function
def import_csv(path):
    """Data import function """
    file = open(path, newline='', encoding="utf-8", errors="ignore")
    reader = csv.reader(file)
    header = next(reader) # exclude the header of the data and store it in variable "header"
    #print(header)
    data = [row for row in reader]
    return data

class video():
    """docstring for video."""
    def __init__(self, data):
        #for key, value in data.items() :
            #print (key, value)
        #self.category = data["category"]
        self.title = data[0].encode('utf-8').decode('utf-8')
        #print(self.title)

        if type(data[2]) == "string": self.id = data[2]
        else: self.id = 0

        if type(data[3]) == "int": self.ratings = int(data[3])
        else:self.ratings = 0

        if type(data[4]) == "int": self.total = int(data[4])
        else:self.ratings = 0

        self.number_of_capital_letter = len(re.findall(r'[A-Z]', self.title))
        self.number_of_exclamation_point = len(re.findall('!', self.title))
        self.number_of_interogation_point = len(re.findall('\\?', self.title))

print("IMPORTING DATA")
data = import_csv('data/titles.csv') #create all the title
videos = [video(data[i]) for i in range(0, len(data))]

#print(data[i][0].encode('utf-8'))
#train_videos = [video(data[i]) for i in range(0, len(data) - 1000)]
#test_videos = [video(data[i]) for i in range(len(data) - 1000, len(data))]
