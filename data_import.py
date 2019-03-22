import numpy as np
import os
import sys
import re
import json
import pandas as pd
from pandas.io.json import json_normalize
import csv
import logging
################################################################################
#Creation and configuration of the logger
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = "log.txt",
                    level = logging.DEBUG, #you can change DEBUG to : INFO,WARNING,ERROR,CRITICAL
                    format = LOG_FORMAT,
                    filemode = 'w')
logger = logging.getLogger()
# here is the diferent level you can use
#logger.debug("lolilol")
#logger.info("lolilol")
#logger.warning("e")
#logger.error("ttt")
#logger.critical("eee")

#Import data function
def import_csv(path):
    """CSV Data import function """
    logger.info("Importing CSV")
    try:
        file = open(path, newline='', encoding="utf-8", errors="ignore")
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        logger.error('Failed to open file', exc_info=True)
    reader = csv.reader(file)
    header = next(reader) # exclude the header of the data and store it in variable "header"
    #print(header)
    data = [row for row in reader if any(row)]
    logger.info("CSV import succefull")
    logger.info(len(data))
    return data

class video():
    """docstring for video."""
    def __init__(self, data):
        #for key, value in data.items() :
            #print (key, value)
        #self.category = data["category"]
        self.title = data[0].encode("utf-8").decode('utf-8','ignore')
        #print(self.title)
        if type(data[2]) == "string": self.id = data[2]
        else: self.id = 0

        try:
            self.ratings = int(data[4])
        except Exception as e:
            self.ratings = 0

        self.number_of_capital_letter = len(re.findall(r'[A-Z]', self.title))
        self.number_of_exclamation_point = len(re.findall('!', self.title))
        self.number_of_interogation_point = len(re.findall('\\?', self.title))

print("Importing data")
data = import_csv('data/titles.csv') #create all the title
videos = [video(data[i]) for i in range(0, len(data))]
