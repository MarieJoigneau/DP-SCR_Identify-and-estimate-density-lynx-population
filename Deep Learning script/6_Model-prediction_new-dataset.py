# -*- coding: utf-8 -*-
"""
PROJET DEEP LEARNING LYNX - Stage M2 2023

Author : Marie Joigneau
Supervisor : Olivier Gimenez

Problematic : Re-identify individual and estimate population density from eurasian lynx population

Before : 
- We have made the deep learning model to re-identify the individuals

This part : Here we use the deep learning model to identify the nearest neighbor in a whole new population.
It is done automatically with the pictures being transferred .
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
import string
import random


# =============================================================================
# =============================================================================
# ======================== FUNCTIONS ==========================================
# =============================================================================
# =============================================================================

"""
Title: predict_reid_df

Description: for each individual in the whole new dataset, return the closest individual in the 128d space

Usage: predict_reid_df(model, images, labels, df,my_eval_distance)

Argument: 
-	model (the deep learning model)
-   images (the pictures to analyze)
-	labels (the identities of the pictures)
-   df (dataframe with information on pictures)
-   my_eval_distance (metric used for the k neigbors)

Details: 
By k-neighbors and the deep learning model, evaluate the k nearest neighbors and 
return dataframe of distances and labels

Values: 
-   reid_dataframe (dataframe)
"""

def predict_reid_df(model, images, labels, df,my_eval_distance):

    # Results of the model prediction :
    # - Extract the 128 dimension vectors (x number of picture) of the embeddings from the test images
    test_features = model.predict(images) 

    # Train a nearest neighbors model with the extracted embeddings.
    neighbors = NearestNeighbors(n_neighbors=11, 
        algorithm='brute',
        metric=my_eval_distance).fit(test_features) 
    
    reid = []

    # For all the embedding vectors 
    for id, feature in enumerate(test_features):

        print("============= THE ID IS ",id,"=============")

        # distances : distances to the k neighbors of the individual n°id (array with 11 values of type float32)
        # - array([[0.        , 0.55303955, 0.56033844, 0.5821051 , 0.586137  , 0.6210177 , 0.65831745, 0.7137499 , 0.8074082 , 0.8232401 , 0.88148504]], dtype=float32)
        # indices : indices of the k neighbors of the individual n°id (array with 11 values of type int64)
        # - array([[ 0,  2,  1,  4, 12, 16,  5, 14, 15, 11, 13]], dtype=int64))
        distances, indices = neighbors.kneighbors([feature])        
        
        reid.append((id, distances[0][1],os.path.basename(df.path.iloc[id]),os.path.basename(labels[indices[0][1]])))
        print("The RE-ID is ",reid)
        
    reid_dataframe = pd.DataFrame(reid, columns=("id", "distance", "pic_name","pic_near"))
    
    print("The Re-ID dataframe is : ")
    print(reid_dataframe)
    
    return(reid_dataframe)




"""
Title: put_together_common

Description: merge sub-vectors which have values in common

Usage: put_together_common(l)

Argument: 
-	l : a vector with sub-vector

Details: 
Find the similar values of the sub-vectors and merge the ones which have values in common

Values: out (a vector with sub-vector)
"""



def put_together_common(l):
    
    # We put the vectors together :
    out = []
    while len(l)>0:
        first, *rest = l
        first = set(first)
    
        lf = -1
        while len(first)>lf:
            lf = len(first)
    
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2
    
        out.append(first)
        l = rest
    
    return(out)



# =============================================================================
# ====== MAIN FUNCTION ========================================================
# =============================================================================  



def prediction_all_new(my_dir_best_model):

    # ======= I/ LOAD THE MODEL ===============================================

    # We load the keras model
    model_saved = tf.keras.models.load_model(my_dir_best_model)
    model_saved.summary()
    
    
   # ======= II/ PREPARE THE DATASET TO EXAMINE ===============================
   # -- 1) The dataframe ------------------------------------------------------

    test_dataset = []
    
    # Iterate over the folders (individuals)
    for indiv in os.listdir(dir_pic):
        for picturename in os.listdir(os.path.join(dir_pic, indiv)):
            test_dataset.append({'individual':indiv, 'path': os.path.join(dir_pic, indiv, picturename)})

    # Prepare a dataframe and preview it
    test_dataset = pd.DataFrame(test_dataset)
    test_dataset.head(10)


    #Let's analyse that train dataset
    print("The shape of the test dataset is ",test_dataset.shape[0]," lines and ",test_dataset.shape[1], "columns") 
    print("The minimal picture number by class is : ",np.min(test_dataset["individual"].value_counts())) # Minimal number of picture for the classes


    # -- 2) The images and labels ---------------------------------------------


    images_test, labels_test = [], []

    # Iterate over the rows of the newly created dataframe
    for (_, row) in tqdm(test_dataset.iterrows()):
        # Read the image from the disk, resize it, and scale the pixel
        # values to [0,1] range
        image = plt.imread(row['path'])
        images_test.append(image)

        # Parse the name of the person
        print(row['individual'])
        labels_test.append(row['path'])

    images_test = np.array(images_test)
    labels_test = np.array(labels_test)
    print(images_test.shape, labels_test.shape)

    # Here the informations
    print("There are ", images_test.shape[0], " pictures, each picture has ", images_test.shape[1], " rows and ", images_test.shape[2], "columns. For each pixel, there are", images_test.shape[3]," values describing them (RGB red green blue)")
    print("Here the labels of the test dataset: ",labels_test)
    
    
    # ======= III/ PREDICT THE NEAREST NEIGHBOR FOR EACH PICTURE ==============
    
    reid_df = predict_reid_df(model_saved, images_test, labels_test,test_dataset,EVAL_DISTANCE)
    
    # ======= IV/ PREPARE THE VECTOR WITH THE SUB GROUPS ======================
    
    # -- 1) The pictures alone ------------------------------------------------
    
    print("-- 1) The pictures alone ------------------------------------------------")
    
    # Here all the pictures with dist > treshold which are alone in the folder
    m_alone =  reid_df[reid_df.distance > TRESHOLD].pic_name.tolist()
    
    # we separate them into several sub-vectors for later
    if len(m_alone) >= 1:
        alone = [[m_alone[0]]]
        for i in range(1,len(m_alone)):
            print(i)
            alone.append([m_alone[i]])
    
    print("The vector with alone pictures is ")
    print(alone)
        
        
    # -- 2) The pictures in group ---------------------------------------------
    print("-- 2) The pictures in group ---------------------------------------------")
    
    
    # we need to put in group the others
    m_group =  reid_df[reid_df.distance <= TRESHOLD]

    # We create a vector with sub-vector of each association
    vec_near = m_group.pic_near.tolist()
    vec_pic = m_group.pic_name.tolist()
    vec_m = []
    for i in range(len(vec_near )):
        vec_m.append([vec_near[i],vec_pic[i]])
    print("the raw vector with grouped pictures is :")
    print(vec_m)
    
    # We merge the sub-vectors with common values
    out = put_together_common(vec_m)
    
    print(out)

    # And convert the list into a vector
    grouped = []
    for i in range(len(out)):
        grouped = grouped + [list(out[i])]
    print("the final vector with grouped pictures is :")
    print(grouped)


    # -- 3) And finally all the pictures in a vector with sub-vectors ---------
    print("-- 3) And finally all the pictures in a vector with sub-vectors ---------")
    
    if len(m_alone) >= 1:
        all_lynx = alone + grouped
    else:
        all_lynx = grouped
    
    print("the final vector with all pictures is ")
    print(all_lynx)
    
    # We copy all the lynx
    for i in range(len(all_lynx)):
        
        print("i is ",i)
        print(all_lynx[i])
        
        # We create the folder of the destination
        name_folder_i = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        os.makedirs(os.path.join(dir_dataset_new,name_folder_i))
        
        # For each lynx in this group :
        for j in range(len(all_lynx[i])):
            
            print("j is ",j)
        
            source = os.path.join(dir_pic,"all",all_lynx[i][j])
            
            # The path of the destination of the picture :
            destination = os.path.join(dir_dataset_new,name_folder_i,all_lynx[i][j])
            
            # Then we copy
            shutil.copy(source,destination) 


# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================

# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

# Metric of the k neighbors :
EVAL_DISTANCE = 'cosine' 
# If you want other options :
#EVAL_DISTANCE = ["l2","euclidean","cosine"]

# Treshold to know if it's a new individual or not
TRESHOLD = 0.56


# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================


# FOLDER PICTURES TO PREDICT : where the pictures to predict are
dir_pic = "D:/deep-learning_re-id_gimenez/6_Model-prediction_new-dataset/dataset_to_predictTEST"
print("dir_pic is ",dir_pic)

# FOLDER DATASET NEW PREDICTED : where the new identities are predicted and put in folders
dir_dataset_new = "D:/deep-learning_re-id_gimenez/6_Model-prediction_new-dataset/new_dataset"
print("The directory of the new dataset pictures is ",dir_dataset_new,"\n")

# BEST MODEL DIRECTORY : Where the best model is
dir_best_model = "D:/deep-learning_re-id_gimenez/2_Model-construction/result_model/300size_angular_EfficientNet"

# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

#disable chained assignments
pd.options.mode.chained_assignment = None 

prediction_all_new(my_dir_best_model=dir_best_model)







    
    
    
    
    
    
    
    
    


