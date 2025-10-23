import sys, os
sys.path.append(os.path.join(os.getcwd(), 'CONFOLD'))

import numpy as np
import pandas as pd
from CONFOLD.foldrm import Classifier

class MyClassifier(Classifier):
        def load_data(self, file, amount=-1):
            data, self.attrs = use_dataframe(file, self.attrs, self.label, self.numeric, amount)
            return data
        

def use_dataframe(file, attrs, label, numeric, amount):
    df = pd.read_csv(file) #get the dataframe

    #split into x and y
    df_x = df[attrs]
    df_y = df[label]
    result = pd.concat([df_x,df_y], axis=1)
    attrs.append(label)
    return result,attrs