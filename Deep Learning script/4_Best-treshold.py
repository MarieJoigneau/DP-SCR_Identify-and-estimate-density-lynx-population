# -*- coding: utf-8 -*-
"""
PROJET DEEP LEARNING LYNX - Internship M2 2023

Author : Marie Joigneau
Supervisor : Olivier Gimenez

Problematic : Re-identify individual and estimate population density from eurasian lynx population

Before : 
- We have made the deep learning model to re-identify the individuals

This part : As we are in an open population, we need to find a treshold. It permits to know for
the nearest neighbor of a picture if it's the same individual or a new one. If distance > treshold,
it's a new one, if distance < treshold, it's the same individual.
We test with differents values of treshold on the test dataset and evaluate the best one with
the True Positive (% of correct predictions of known individuals) and the True Negative (% of
correct predictions of unknown individuals)
"""


# =============================================================================
# =============================================================================
# ======================== PACKAGES ===========================================
# =============================================================================
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import tensorflow
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# =============================================================================
# =============================================================================
# ======================== FUNCTIONS ==========================================
# =============================================================================
# =============================================================================

"""
Title: folder_1_pic

Description: give the individuals with 1 pictures

Usage: folder_1_pic(my_dir_pop)

Argument: 
-	my_dir_pop (directory of all the pictures)

Details: 
Counts the pictures of each individual, and if it's 1, the individual name is 
added in the vector ind_1_picture

Values: 
-   ind_1_picture (vector of all the individuals having 1 pictures in this directory)
"""

def folder_1_pic(my_dir_pop):
    
    # Individual list :
    list_ind = os.listdir(my_dir_pop) 
    print("Here the list of all the individuals : ",list_ind)
    
    # Initialisation of the 1-file animal vector
    ind_1_picture = [] 

    for ind in list_ind: # For each animal
        
        dir_ind = os.path.join(my_dir_pop,ind) # We go in the folder of the individual (ind)

        # We count all the files in the folder  :
        initial_count = 0 # We initialise
        for path in os.listdir(dir_ind):
            # If we have 1 picture we add 1 in the count
            if os.path.isfile(os.path.join(dir_ind, path)):
                initial_count += 1

        # If there is only 1 picture, we put the name of the folder (the name of the animal) in the vector
        if initial_count == 1 :
            ind_1_picture = ind_1_picture + [ind]
    
    # Ouput : the vector of the animal names
    return(ind_1_picture)


"""
Title: predict_class_df_treshold

Description: return the dataframes of re-identification or new-identification 
of the nearest neighbors for each picture

Usage: predict_class_df_treshold(model, images, train_labels, labels,df,my_eval_distance,ind_new_id)

Argument: 
-	model (the deep learning model)
-   images (the pictures to analyze)
-   labels (the identities of the pictures)
-   df (the dataframe with all the informations on the pictures)
-   my_eval_distance (metric chosen of the k-neighbors)
-   ind_new_id (individuals with 1 picture)

Details: 
Return the dataframes of re-identification or new-identification of the nearest 
neighbors for each picture

Values: 
-   reid_dataframe : dataframe for the individuals to re-identify
-   newid_dataframe : dataframe for the new individuals
"""


def predict_class_df_treshold(model, images, labels,df,my_eval_distance,ind_new_id):

    # Results of the model prediction :
    # - Extract the 128 dimension vectors (x number of picture) of the embeddings from the test images
    test_features = model.predict(images) 

    # Train a nearest neighbors model with the extracted embeddings. Here we are limiting it to 11 neighbors only
    neighbors = NearestNeighbors(n_neighbors=11, 
        algorithm='brute',
        metric=my_eval_distance).fit(test_features) 
    
    reid = []
    newid = []

    # For all the embedding vectors 
    for id, feature in enumerate(test_features):

        print("============= THE ID IS ",id,"=============")

        # distances : distances to the k neighbors of the individual n°id (array with 11 values of type float32)
        # - array([[0.        , 0.55303955, 0.56033844, 0.5821051 , 0.586137  , 0.6210177 , 0.65831745, 0.7137499 , 0.8074082 , 0.8232401 , 0.88148504]], dtype=float32)
        # indices : indices of the k neighbors of the individual n°id (array with 11 values of type int64)
        # - array([[ 0,  2,  1,  4, 12, 16,  5, 14, 15, 11, 13]], dtype=int64))
        distances, indices = neighbors.kneighbors([feature])

        # Here the labels of the k neighbours next to the test individual :
        # - k-1 = 10 individuals here
        # - ['1376', '1376', '1376', 'Minus', 'Minus', '1376', 'Minus', 'Minus', 'Letty', 'Minus']
        similar_labels = [labels[indices[0][i]] for i in range(1, 11)]
        print("The labels of the individual is ",labels[indices[0][0]])
        
        # If it's an individual already known (>1 picture)
        if labels[id] not in ind_new_id:
            
            # we initialise the indice of the k nearest neighbors with same identity
            i_top = 0

            for id2, similar_label in enumerate(similar_labels):
                
                # In the k neighbors of the individual n°id, we find the k-th neighbors that verify the initial label
                if similar_label == labels[id]:
                    # In the result :
                    # - id = the id of the test individual
                    # - id2+1 
                    #      => we put +1 because id2=0 is the test individual
                    #      => the k-th closest neighbors that verify the identity of the test individual
                    # - distances[0][id2+1] = We save the distance of the closest neighbors 
                    result = (id, distances[0][id2+1], id2+1,os.path.basename(df.path.iloc[id]),labels[id],distances[0][1])
                    print("(RE-ID) The distances of the closest right label is ",distances[0][id2+1])
                    break
                
                # if we don't manage to find an image of the same individual in the top 11
                if i_top == (len(similar_labels)-1): 
                    print("i_top = 11")
                    result = (id, 100, 11, os.path.basename(df.path.iloc[id]),labels[id],distances[0][1])
                i_top = i_top +1
                print("i_top is ",i_top)
                
            # The result goes in the reid_df
            reid.append(result)
        
        # If it's a new individual (picture = 1)
        if labels[id] in ind_new_id:
            # If it's not in the train labels :
            # - The result goes in the newid_df
            # - We save the distance of the closest neighbors
            newid.append((id, distances[0][1],os.path.basename(df.path.iloc[id]),labels[id]))
        print("(NEW-ID) The distance of the closest label is ",distances[0][1])
    
    reid_dataframe = pd.DataFrame(reid, columns=("id", "distance", "top","pic_name","label","top_dist"))
    newid_dataframe = pd.DataFrame(newid, columns=("id", "distance","pic_name","label"))
    
    print("The re-ID dataframe is : ")
    print(reid_dataframe)
    print("")
    print("The new-ID dataframe is : ")
    print(newid_dataframe)
    
    return(reid_dataframe,newid_dataframe)


# =============================================================================
# ====== MAIN FUNCTION ========================================================
# =============================================================================    


"""
Title: best_treshold

Description: return evaluation metrics depending of the treshold

Usage: best_treshold(my_dir_split,my_dir_best_model,my_treshold_list,my_dir_treshold,my_folder_test)

Argument: 
-	my_dir_split : directory where all the splitted pictures are
-   my_dir_best_model : directory of the model used
-   my_treshold_list : list of all the treshold values to test
-   my_dir_treshold : directory where the csv is saved
-   my_folder_test : sub-folder of my_dir_split with the pictures to test

Details: 
Return evaluation metrics (True Positive and True Negative) depending of 
the treshold in a csv

Values: /
"""


def best_treshold(my_dir_split,my_dir_best_model,my_treshold_list,my_dir_treshold,my_folder_test):
    
    # ======= I/ LOAD THE MODEL ===============================================

    model_saved = tf.keras.models.load_model(my_dir_best_model)
    model_saved.summary()
    
   # ======= II/ PREPARE THE TEST DATASET =====================================
   # -- 1) The dataframe -----------------------------------------------------

    test_dataset = []
    
    # Iterate over the folders (individuals)
    for indiv in os.listdir(os.path.join(my_dir_split, my_folder_test)):
        for picturename in os.listdir(os.path.join(my_dir_split, my_folder_test, indiv)):
            test_dataset.append({'individual':indiv, 'path': os.path.join(my_dir_split, my_folder_test, indiv, picturename)})

    # Prepare a dataframe and preview it
    test_dataset = pd.DataFrame(test_dataset)
    test_dataset.head(10)


    #Let's analyse that train dataset
    print("The shape of the test dataset is ",test_dataset.shape[0]," lines and ",test_dataset.shape[1], "columns") 
    print("The minimal picture number by class is : ",np.min(test_dataset["individual"].value_counts())) # Minimal number of picture for the classes

    nb_individual = len(os.listdir(os.path.join(my_dir_split, my_folder_test)))
    print("There are ",nb_individual," individuals in the test dataset")

    # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python (get unique value)
    ind_test_unique=list(test_dataset["individual"])
    ind_test_unique=set(ind_test_unique)
    ind_test_unique=list(ind_test_unique)
    print("Here the individuals of the test dataset : ",ind_test_unique)

    print(test_dataset.individual.value_counts()) # Show number of picture by individual



    # -- 2) The test images and labels ----------------------------------------
    
    images_test, labels_test = [], []

    # Iterate over the rows of the newly created dataframe
    for (_, row) in tqdm(test_dataset.iterrows()):
        # Read the image from the disk, resize it, and scale the pixel
        # values to [0,1] range
        image = plt.imread(row['path'])
        images_test.append(image)

        # Parse the name of the person
        print(row['individual'])
        labels_test.append(row['individual'])

    images_test = np.array(images_test)
    labels_test = np.array(labels_test)
    print(images_test.shape, labels_test.shape)

    # Here the informations
    print("There are ", images_test.shape[0], " pictures, each picture has ", images_test.shape[1], " rows and ", images_test.shape[2], "columns. For each pixel, there are", images_test.shape[3]," values describing them (RGB red green blue)")
    print("Here the labels of the test dataset: ",labels_test)
    
    # ======= III/ TOP-1-ACCURACY = TRUE POSITIVE =============================
    
    # We put the ID of all the unknown individuals
    ind_new_id = folder_1_pic(os.path.join(my_dir_split,my_folder_test))[0]
    
    # We take the 2 dataframe for new and re-id
    reid_df, newid_df = predict_class_df_treshold(model_saved, images_test, labels_test,test_dataset,EVAL_DISTANCE,ind_new_id)
    print(newid_df)
    print(reid_df)
     
    result = []
     
    for my_treshold_ind in my_treshold_list:

        # We see how many are in top-1 and under the treshold
        c = reid_df[(reid_df.top == 1) & (reid_df.distance <= my_treshold_ind)]
        
        top_1_acc = len(c)/len(reid_df)
        print("top-1-acc is ",top_1_acc)
        
        # ======= IV/ TRUE NEGATIVE ===========================================

        # We see how many new id are over the treshold
        c = newid_df[newid_df.distance >= my_treshold_ind]

        TN = len(c)/len(newid_df)
        print("TN is ",TN)

        
        # ======= V/ WE SAVE THE RESULTS ======================================
        
        # We group all the results
        result_FP_TP = (my_treshold_ind,top_1_acc,TN)
        result.append(result_FP_TP)
        print("result is")
        print(result)
        
    # We convert the results into a dataframe
    result_df = pd.DataFrame(result, columns=("treshold","top_1_acc","TN"))
    print("The final dataframe with the results is :")
    print(result_df)
    
    # We save the results in a csv
    path_save = os.path.join(my_dir_treshold,"result_treshold.csv")
    result_df.to_csv(path_save, sep=';')

   

# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

# If you want to choose a single model :
EVAL_DISTANCE = 'cosine' 
# If you want to search the best model by modifying:
#EVAL_DISTANCE = ["l2","euclidean","cosine"]

TRESHOLD = np.arange(0, 1.05, 0.01)

# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# FOLDER SPLIT : Folder with split images which will be used for the deep-learning)
dir_split = "D:/deep-learning_re-id_gimenez/1_Pre-processing/dataset_split_224"
print("The directory of the splitted pictures is ",dir_split,"\n")

# FOLDER DATASET TEST : subfolder of DIR-SPLIT where the pictures are
folder_test = "test_pour-treshold-avec-1-photo"

# TRESHOLD DIRECTORY : Where we will do all the treshold stuff
# - prediction of the class of the individuals
# - result of the best treshold in a csv
dir_treshold = "D:/deep-learning_re-id_gimenez/4_Best-treshold"

# FOLDER PREDICTION TEST : where we test different prediction depending of the treshold
dir_pred_test = "D:/deep-learning_re-id_gimenez/4_Best-treshold/dataset_treshold"
print("dir_pred_test is ",dir_pred_test)

# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

#disable chained assignments
pd.options.mode.chained_assignment = None 

best_treshold(my_dir_split=dir_split,my_dir_pred_test=dir_pred_test,my_treshold_list=TRESHOLD,my_dir_treshold=dir_treshold,my_folder_test=folder_test)









