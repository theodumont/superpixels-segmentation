# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser

from math import *
import numpy as np
import json

home = expanduser("~")


if __name__ == '__main__':

    results_path = "./ref200/"
    with open(results_path + "result.json") as f:
        data = json.load(f)

    recall = data['recall']
    recall2 = data['recall2']
    undersegmentation = data['undersegmentation']
    undersegmentationNP = data['undersegmentationNP']
    precision = data['precision']
    compactness = data['compactness']

    print("Recall: " + str(np.mean(recall)))
    # print("Recall2: " + str(np.mean(recall2)))
    # print("Precision: " + str(np.mean(precision)))
    print("Undersegmentation: " + str(np.mean(undersegmentation)))
    # print("Undersegmentation (NP): " + str(np.mean(undersegmentationNP)))
    print("Compactness: " + str(np.mean(compactness)))
