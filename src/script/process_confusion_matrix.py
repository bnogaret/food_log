#! /usr/bin/env python3
import os

import pandas as pd
import numpy as np

import init_path

from thesis_lib.uecfood import get_name_and_category
from thesis_lib.io import load_object, PATH_TO_PICKLE_SAVE

import constants as const


PATH_TO_DATABASE = "/scratch/s242635/miniconda3/envs/thesis/project/data/UECFOOD256/"
CATEGORY_FILE = PATH_TO_DATABASE + "category.txt"

def main():
    df = get_name_and_category(CATEGORY_FILE)

    print(df.shape)
    print(df.dtypes)
    print(df.describe())
    print(df.head())
    print(df._category.unique())

    n = df['_name'].to_dict()
    print(n)
    
    list_file_cm = ['cm_bow.pk', 'cm_DecisionTreeClassifier.pk', 'cm_GaussianNB.pk',
                    'cm_KNeighborsClassifier.pk', 'cm_lbp_hsv.pk', 'cm_LinearSVC.pk', 
                    'cm_RandomForestClassifier.pk', 'cm_SGDClassifier.pk']
    
    eps = 1e-5 # to avoid 0.0 divide
    
    for file_cm in list_file_cm[:1]:
        filename = PATH_TO_PICKLE_SAVE + "/" + file_cm
        print(filename)
        
        if os.path.isfile(filename):
            cm = load_object(file_cm)
            # print(cm)
            
            # Divide by the number of elements
            cm_scaled = cm / (cm.sum(axis=1, keepdims=True) + eps)

            diag = np.diagonal(cm_scaled)
            
            # Ten most accuracy score
            for m in np.argsort(diag)[-10:]:
                print(n[m + 1], diag[m])
            
            # Ten least accuracy score
            for l in np.argsort(diag)[:10]:
                print(n[l + 1], diag[l])
            
            # 10 most confused
            np.fill_diagonal(cm_scaled, 0.0)
            for t in range(0, 10):
                i,j = np.unravel_index(cm_scaled.argmax(), cm_scaled.shape)
                print(i, j, n[i + 1], "||", n[j + 1], cm_scaled[i, j])
                cm_scaled[i, j] = 0.0

if __name__ == "__main__":
    main()


