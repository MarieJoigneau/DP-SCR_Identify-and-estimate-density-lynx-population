# -*- coding: utf-8 -*-

"""
PROJET DEEP LEARNING LYNX - Internship M2 2023

Author : Marie Joigneau
Supervisor : Olivier Gimenez

Problematic : Re-identify individual and estimate population density from eurasian lynx population

Before : 
- We have made the deep learning model to re-identify the individuals

This part : Here we use the deep learning model to identify the nearest neighbor in a population.
It is done automatically with a csv at the end recording what the IA has done.
"""

# =============================================================================
# =============================================================================
# ======================== PACKAGES ===========================================
# =============================================================================
# =============================================================================


import numpy as np
import os
import tensorflow
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import string
import random

# =============================================================================
# =============================================================================
# ======================== FUNCTIONS ==========================================
# =============================================================================
# =============================================================================



"""
Title: predict_class

Description: return labels and distances of the individuals the closest an individual in the 128d space

Usage: predict_class(my_model_saved, my_images_pred, my_labels_pred,nb,my_number_0new)

Argument: 
-	my_model_saved (the deep learning model)
-   my_images_pred (the pictures to analyze)
-	my_labels_pred (the identities of the pictures)
-   nb (the ID of the individual to predict)
-   my_number_0new (the total of individuals to predict)

Details: 
By k-neighbors and the deep learning model, evaluate the k nearest neighbors and 
return distances and labels

Values: 
-   label_top5 (the identities of the 5 nearest individuals)
-   dist_top5 (the distances of the 5 nearest individuals)
"""



def predict_class(my_model_saved, my_images_pred, my_labels_pred,nb,my_number_0new):
    
    print("============= THE ID IS ",nb,"=============")
    
    # We put away the other new pictures not predicted yet :
    idx_predict = [j for j in range(nb+1)] + [i for i in range(my_number_0new,len(my_images_pred))]
    print("The index in the predicted dataset are ")
    print(idx_predict)
    
    # And we do the same for the labels :
    my_labels_pred = my_labels_pred[idx_predict]
    print("The labels in the predicted dataset are ")
    print(my_labels_pred)
    
    # Results of the model prediction :
    # - Extract the 128 dimension vectors (x number of picture) of the embeddings from the test images
    test_features = my_model_saved.predict(my_images_pred[idx_predict]) 
    
    # Train a nearest neighbors model with the extracted embeddings. Here we are limiting it to 11 neighbors only
    neighbors = NearestNeighbors(n_neighbors=11, 
        algorithm='brute',
        metric=EVAL_DISTANCE).fit(test_features)

    # distances : distances to the k neighbors of the individual n°id (array with 11 values of type float32)
    # - array([[0.        , 0.55303955, 0.56033844, 0.5821051 , 0.586137  , 0.6210177 , 0.65831745, 0.7137499 , 0.8074082 , 0.8232401 , 0.88148504]], dtype=float32)
    # indices : indices of the k neighbors of the individual n°id (array with 11 values of type int64)
    # - array([[ 0,  2,  1,  4, 12, 16,  5, 14, 15, 11, 13]], dtype=int64))
    distances, indices = neighbors.kneighbors([test_features[nb]])
        
    print("The nearest distances are ")
    print(distances)
    print("The nearest indices are ")
    print(indices)

    print("TOP 5 ARE")
    label_top5 = [my_labels_pred[indices[0][i]] for i in range(1, 6)]
    print(label_top5)
    dist_top5 = [distances[0][i] for i in range(1,6)]
    print(dist_top5)
    
    return(label_top5,dist_top5)



"""
Title: predict_class_all_ind

Description: return labels and distances of all the individuals to predict for the population in the 128d space

Usage: predict_class_all_ind(my_predict_dataset,my_dir_predict,my_model_saved,my_images_pred,my_labels_pred,my_number_0new):

Argument: 
-	my_model_saved (the deep learning model)
-   my_images_pred (the pictures to analyze)
-	my_labels_pred (the identities of the pictures)
-   nb (the ID of the individual to predict)
-   my_number_0new (the total of individuals to predict)

Details: 
By using the predict_class function on all the individuals to predict, we obtain a dataframe recording the results of the closest ones

Values: 
-   result_dataframe (dataframe with for each individual the nearest ones)
"""



def predict_class_all_ind(my_predict_dataset,my_dir_predict,my_model_saved,my_images_pred,my_labels_pred,my_number_0new):
    
    result = []
    
    for k in range(0,my_number_0new):
        
        # ======= I/ THE PREDICTION ===========================================
        
        # We predict
        label_top,dist_top = predict_class(my_model_saved, my_images_pred, my_labels_pred,k,my_number_0new)

        top1_ind, top1_dist = label_top[0],dist_top[0]
        print("The label of the closest individual is ",top1_ind, " with a distance of ",top1_dist)
        
        # If the individual is close enough to the closest one :
        if (top1_dist<TRESHOLD):
            print("It's a re-ID")
            new_or_not = "individu_connu"
            new = 0
            new_or_not_name = str(top1_ind)
        # If he is too far away :
        else:
            # We ask the name of the new class
            print("It's a new individual")
            new_or_not = "nouvel_individu"
            new = 1
            new_or_not_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
            print("Its new name is ",new_or_not_name)
        
        result.append((os.path.basename(my_predict_dataset.path[k]),new_or_not,new_or_not_name,label_top[0],dist_top[0],label_top[1],dist_top[1],label_top[2],dist_top[2],label_top[3],dist_top[3],label_top[4],dist_top[4]))
        
        print("The result until the ",k," individual is")
        print(result)
        
        # ======= II/ WE COPY THE PICTURE IN THE CORRECT PREDICTED FOLDER =====
        
        # We first change the label of the individual for the futur predictions of the remaining individuals
        my_labels_pred[k] = str(new_or_not_name)
        
        # And we create the new folder if necessary
        if new == 1:
            new_folder = os.path.join(my_dir_predict, new_or_not_name)
            print("Here the new folder : ",new_folder)
            os.makedirs(new_folder)
         
        # The path of the source of the picture :
        source = my_predict_dataset.path[k] # For example 'D:/deep-learning_re-id_gimenez\dataset_predict\0new\1376_1.jpg'
        print("The source is ", source)
    
        # The name of the picture (in the path of the source of the picture):
        pic_name = os.path.basename(my_predict_dataset.path[k]) # For example '1376_1.jpg' in 'D:/deep-learning_re-id_gimenez\dataset_predict\0new\1376_1.jpg'
        print("The picture name is ",pic_name)
    
        # The path of the destination of the picture :
        destination = os.path.join(my_dir_predict, str(new_or_not_name),pic_name) # For example 'D:/deep-learning_re-id_gimenez\dataset_predict\1376\1376_1.jpg'
        print("The destination is ",destination)
    
        # Then we copy
        shutil.copyfile(source,destination) 
        
    result_dataframe = pd.DataFrame(result, columns=("image", "individu_attribue","connu_ou_pas","individu1", "distance1", "individu2", "distance2", "individu3", "distance3", "individu4", "distance4", "individu5", "distance5"))
    
    print("The final result is ")
    print(result_dataframe)
    
    return(result_dataframe)




# =============================================================================
# === MAIN FUNCTION ===========================================================
# =============================================================================


"""
Title: predict_already_dataset

Description: return labels and distances of all the individuals to predict for the population in the 128d space

Usage: predict_already_dataset(my_dir_predict)

Argument: 
-	my_dir_predict : directory of the entire population with individuals to predict

Details: 
In the prediction dataset, we have chosen :
 - To put the new individual to predict in the folder 0new. The 0new individuals can be re-ID or new.
 - To help in the class determination by k neigbors with other pictures of individual
The prediction dataset is the population we use as reference and the individuals we want to predict  
We move the pictures to predict in the right folder (of the identity predicted) 
or in a new one (if it's a new individual), and write that in a csv

Values: 
-   result_dataframe (dataframe with for each individual the nearest ones)
"""


def predict_already_dataset(my_dir_predict):

    # ======= I/ PREPARE THE DATASET FOR PREDICTION ===========================
    # -- 1) The dataframe -----------------------------------------------------
    
    predict_dataset = []

    # Iterate over the folders (individuals)
    for indiv in os.listdir(my_dir_predict):
        for picturename in os.listdir(os.path.join(my_dir_predict, indiv)):
            predict_dataset.append({'individual':indiv, 'path': os.path.join(my_dir_predict, indiv, picturename)})

    # Prepare a dataframe and preview it
    predict_dataset = pd.DataFrame(predict_dataset)
    predict_dataset.head(10)
    
    # -- 2) Analysis of the dataframe -----------------------------------------

    #Let's analyse that predict dataset
    print("The shape of the prediction dataset is ",predict_dataset.shape[0]," lines and ",predict_dataset.shape[1], "columns") 
    print("The minimal picture number by class is : ",np.min(predict_dataset["individual"].value_counts())) # Minimal number of picture for the classes

    nb_individual = len(os.listdir(dir_predict))
    print("There are ",nb_individual," individuals in the prediction dataset")

    # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python (get unique value)
    ind_predict_unique=list(predict_dataset["individual"])
    ind_predict_unique=set(ind_predict_unique)
    ind_predict_unique=list(ind_predict_unique)
    print("Here the individuals of the prediction dataset : ",ind_predict_unique)

    print(predict_dataset.individual.value_counts()) # Show number of picture by individual
    
    # Number of individuals to predict (not the entire dataset)
    number_0new = len(os.listdir(dir_new_ind))

    # -- 3) The prediction images and labels ----------------------------------

    images_pred, labels_pred = [], []

    # Iterate over the rows of the newly created dataframe
    for (_, row) in tqdm(predict_dataset.iterrows()):
        image_pred = plt.imread(row['path'])
        images_pred.append(image_pred)
        # Parse the name of the individual
        labels_pred.append(row['individual'])

    images_pred = np.array(images_pred)
    labels_pred = np.array(labels_pred)

    # Here we have taken the pixel info on the test dataset pictures
    print("There are ", images_pred.shape[0], " pictures, each picture has ", images_pred.shape[1], " rows and ", images_pred.shape[2], "columns. For each pixel, there are", images_pred.shape[3]," values describing them (RGB red green blue)")

    # ======= II/ CALL THE DEEP LEARNING MODEL ================================

    # We load the keras model
    model_saved = tf.keras.models.load_model(dir_best_model)
    model_saved.summary()


     # ======= III/ PREDICTION OF THE PICTURES ================================

    df = predict_class_all_ind(predict_dataset,my_dir_predict,model_saved,images_pred,labels_pred,my_number_0new=number_0new)
    
    # We save the dataframe
    df.to_csv(dir_results, sep=';',index=False)



# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

# KNN :
# - cosine = Compute cosine distance between samples in X and Y (Cosine distance is defined as 1.0 minus the cosine similarity)
# (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html#sklearn.metrics.pairwise.cosine_distances)
# - 'euclidian' = d (P, Q) = √ ((XQ - XP) ^ 2 + (YQ - YP) 
EVAL_DISTANCE = 'cosine'


# Treshold between 2 individuals to know if they are the same or different
TRESHOLD = 0.56


# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# Folder where the best model is
dir_best_model = "D:/deep-learning_re-id_gimenez/2_Model-construction/result_model/300size_angular_EfficientNet"

# Folder used for the prediction
dir_predict = 'D:/deep-learning_re-id_gimenez/5_Model-prediction_already-dataset/dataset'
print("The path with all the individuals for the prediction is ",dir_predict)

# We put here all the new pictures to predict :
dir_new_ind = os.path.join(dir_predict,"0new")
print("The directory of the pictures to predict is ",dir_new_ind)

dir_metadata = "D:/deep-learning_re-id_gimenez/5_Model-prediction_already-dataset/lynx_metadata.csv"
print("The directory of the predicted csv is ",dir_metadata)

dir_results = "D:/deep-learning_re-id_gimenez/5_Model-prediction_already-dataset/resultats_k-voisins_auto.csv"


# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

predict_already_dataset(my_dir_predict=dir_predict)



