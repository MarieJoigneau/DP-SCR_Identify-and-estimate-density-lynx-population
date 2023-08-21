# -*- coding: utf-8 -*-
"""
PROJET DEEP LEARNING LYNX - Internship M2 2023

Author : Marie Joigneau
Supervisor : Olivier Gimenez

Problematic : Re-identify individual and estimate population density from eurasian lynx population

This part : Here we will prepare the pictures to construct and test deep learning models
More particulary, we want to change the minimal picture by individual
"""

# =============================================================================
# =============================================================================
# ======================== PACKAGES ===========================================
# =============================================================================
# =============================================================================

import pandas as pd
import os
from distutils.dir_util import copy_tree


# =============================================================================
# =============================================================================
# ======================== FUNCTIONS ==========================================
# =============================================================================
# =============================================================================


"""
Title: pic_by_ind

Description: gives the number of pictures by individual

Usage: pic_by_ind(my_dir_pic)

Argument: 
-	my_dir_pic : directory of all the pictures

Details: 
Give the number of pictures by individual

Values: 
-   df : dataframe recording the number of pictures by individual
"""


def pic_by_ind(my_dir_pic):
    
    # Individual list :
    list_ind = os.listdir(my_dir_pic) 
    print("Here the list of all the individuals : ",list_ind)
    
    # Initialisation of the result vector
    result = [] 
    
    for ind in list_ind: # For each animal
        print("the individual is ",ind,"---------------")
        
        dir_ind = os.path.join(my_dir_pic,ind) # We go in the folder of the individual (ind)
        print(dir_ind)
        nb_pic = len(os.listdir(dir_ind))
        print(nb_pic)
        
        nb_pic_ind = (ind,nb_pic)
        print(nb_pic_ind)
     
        result.append(nb_pic_ind)
        
        print("Result is ",result)
    
    df = pd.DataFrame(result, columns=("ind", "nb_pic"))
    
    return(df)


"""
Title: change_min_pic_test_dataset

Description: change the minimal number of picture by individual in a dataset

Usage: change_min_pic_test_dataset(my_dir_split,my_fold_test,my_df,my_min_pic_list)

Argument: 
-	my_dir_split : directory of all the splitted pictures
-   my_fold_test : folder of the pictures to filter
-   my_df : dataframe of the number of pictures by individual
-   my_min_pic_list : list of the number of pictures to have in minimum by individual

Details: 
Change the minimal number of picture by individual in a dataset by deleting the folders
of individual with not a sufficient amount

Values: /
"""


def change_min_pic_test_dataset(my_dir_split,my_fold_test,my_df,my_min_pic_list):
    
    for my_min_pic in my_min_pic_list:
    
        print("The minimal number of pictures by individual is ",my_min_pic)
        
        df_chosen = my_df[my_df.nb_pic>=my_min_pic]    
        print("df_chosen is ")
        print(df_chosen)
        
        new_fold_test = str(my_fold_test) + str("_min_pic") + str(my_min_pic)
        print("The new folder is ",new_fold_test)
        os.makedirs(os.path.join(my_dir_split,new_fold_test))
        
        for ind in df_chosen.ind:
            
            print("We copy the ind ",ind)
            
            source = os.path.join(my_dir_split,my_fold_test,ind)
            destination = os.path.join(my_dir_split,new_fold_test,ind)
            
            os.makedirs(destination)
            copy_tree(source, destination)


# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

# We want to create several test dataset, depending of the minimal number we want
# The initial one is 2 by individual.
# The other are  : 5,8,10,12,15,20 pictures at least by individual
MIN_PIC = [5,8,10,12,15,20]


# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# DIR_SPLIT : directory of all the splitted pictures
dir_split="D:/deep-learning_re-id_gimenez/1_Pre-processing"

# DIR_PIC_CHANGE : directory of the category of pictures to copy
dir_pic_change = "D:/deep-learning_re-id_gimenez/1_Pre-processing/dataset_split_260"

# FOLD_TEST : folder to copy
fold_test = "dataset_split_260/dataset_ready"

# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

# We have the dataframe with the number of pictures by individual in the folder
df_pic_ind = pic_by_ind(dir_pic_change)

# We copy the individuals depending of the number of their pictures
change_min_pic_test_dataset(my_dir_split=dir_split, my_fold_test=fold_test, my_df=df_pic_ind,my_min_pic_list=MIN_PIC)
