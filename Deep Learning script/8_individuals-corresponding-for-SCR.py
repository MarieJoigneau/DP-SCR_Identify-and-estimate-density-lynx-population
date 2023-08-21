# -*- coding: utf-8 -*-
"""
PROJET DEEP LEARNING LYNX - Internship M2 2023

Author : Marie Joigneau
Supervisor : Olivier Gimenez

Problematic : Re-identify individual and estimate population density from eurasian lynx population

Before : 
- We have made the deep learning model to re-identify the individuals
- We have applied it to a new population where we want to estime the density
- We want to compare with the human estimation for the Spatial Capture Recapture model

This part : Here we want a csv with the pictures and their predicted folders
"""


# =============================================================================
# =============================================================================
# ======================== PACKAGES ===========================================
# =============================================================================
# =============================================================================

import pandas as pd
import os


# =============================================================================
# =============================================================================
# ======================== FONCTIONS ==========================================
# =============================================================================
# =============================================================================

"""
Title: link_pic_folder

Description: create a dataframe which links pictures and folder

Usage: link_pic_folder(my_dir_data)

Argument: 
-	my_dir_data (directory of the folders containing the pictures)

Details: 
For each file, write in the dataframe the folder attributed

Values: df (dataframe)
"""

def link_pic_folder(my_dir_data):
    
    # We initialize the data
    data = []
    
    print(os.listdir(my_dir_data))
    
    # For each folder
    for folder in sorted(os.listdir(my_dir_data)):
        # For each file
        for file in sorted(os.listdir(os.path.join(my_dir_data,folder))):
            # We add the correspondance
            data.append((folder, file))
    
    # And we regroup everything in a dataframe
    df = pd.DataFrame(data, columns=['folder', 'file'])
    
    print ("The dataframe we have obtained is ")
    print(df)
    
    return(df)


# =============================================================================
# ====== MAIN FUNCTION ========================================================
# =============================================================================  


def individual_corresponding(dir_data):
    
    # We create the dataframe to do the correpondance
    df = link_pic_folder(dir_data)

    # And we save the dataframe
    path_save = "D:/deep-learning_re-id_gimenez/8_Individual-corresponding/corresponding_individuals.csv"
    df.to_csv(path_save, sep=';', index=False)


# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# DATASET NEW DIRECTORY : Where we have the entire new dataset
dir_dataset_new = "D:/deep-learning_re-id_gimenez/6_Model-prediction_new-dataset/new_dataset"
print("The directory of the new dataset pictures is ",dir_dataset_new,"\n")

# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

individual_corresponding(dir_dataset_new)



