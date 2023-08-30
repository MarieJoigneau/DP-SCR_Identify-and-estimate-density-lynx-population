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

This part : Here we want to know in a csv the distribution of the picture knowing 
their identity to analyse how the model work
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
import tensorflow
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# =============================================================================
# =============================================================================
# ======================== FONCTIONS ==========================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== FUNCTIONS ============================================================
# =============================================================================

"""
Title: evaluate_all_return_csv

Description: return a csv of the individual the closest for the population in the 128d space

Usage: evaluate_all_return_csv(model, images, labels, df,my_eval_distance)

Argument: 
-	model (the deep learning model)
-	images (the pictures to analyze)
-	labels (the identities of the pictures)
-	df (the dataframe containing the information about each picture)
-   my_eval_distance (the treshold chosen)

Details: 
By k-neighbors and the deep learning model, evaluate the k nearest neighbors and 
return a csv with the result

Values: reid_dataframe (dataframe)
"""

def evaluate_all_return_csv(model, images, labels, df,my_eval_distance):

    # Results of the model prediction :
    # It extract the 128 dimension vectors (x number of picture) of the embeddings from the test images
    test_features = model.predict(images) 

    # Train a nearest neighbors model with the extracted embeddings. Here we are limiting it to 11 neighbors only
    neighbors = NearestNeighbors(n_neighbors=11, 
        algorithm='brute',
        metric=my_eval_distance).fit(test_features) 
    
    # We initiate the re-identification dataframe
    reid = []

    # For all the embedding vectors predicted we find the closest :
    for id, feature in enumerate(test_features):
        
        # distances : the k nearest distances
        # -> array([[0.        , 0.55303955, 0.56033844, 0.5821051 , 0.586137  , 0.6210177 , 0.65831745, 0.7137499 , 0.8074082 , 0.8232401 , 0.88148504]], dtype=float32)
        # indices : the k indices of the nearest individuals
        # -> array([[ 0,  2,  1,  4, 12, 16,  5, 14, 15, 11, 13]], dtype=int64))
        distances, indices = neighbors.kneighbors([feature])
        
        # We add the result in the vector for the final dataframe
        # - id : identification of the individual (0,1,2...) 
        # - distances[0][1] : distance of the nearest neighbor
        # - os.path.basename(df.path.iloc[id]) : identity of the individual = pic_name
        # - os.path.basename(labels[indices[0][1]]) : identity if the nearest neighbor = pic_near
        reid.append((id, distances[0][1],os.path.basename(df.path.iloc[id]),os.path.basename(labels[indices[0][1]])))
    
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


"""
Title: distribution

Description: create a dataframe which show for each picture if it is alone and if it is well classed

Usage: distribution(my_dir_split,my_dir_best_model,my_folder_test)

Argument: 
-	my_dir_split : directory where all the splitted pictures are
-   my_dir_best_model : directory of the model
-   my_folder_test : name of the folder with the picture to examine

Details: 
With a model and prepared pictures, we predict the nearest neighbor for each picture,
then depending of the treshold we put them in groups and count how many are well classed or not, 
and the ones alone

Values: /
"""



def distribution(my_dir_split,my_dir_best_model,my_folder_test):

    # ======= I/ LOAD THE MODEL ===============================================

    # We load the keras model
    model_saved = tf.keras.models.load_model(my_dir_best_model)
    model_saved.summary()
    
    
   # ======= II/ PREPARE THE DATASET TO EXAMINE ===============================
   # -- 1) The dataframe ------------------------------------------------------

    test_dataset = []
    
    # Iterate over the folders (individuals)
    for indiv in os.listdir(os.path.join(my_dir_split, my_folder_test)):
        for picturename in os.listdir(os.path.join(my_dir_split, my_folder_test, indiv)):
            test_dataset.append({'individual':indiv, 'path': os.path.join(my_dir_split, my_folder_test, indiv, picturename)})

    # Prepare a dataframe and preview it
    test_dataset = pd.DataFrame(test_dataset)

    #Let's analyse that dataset
    print("The shape of the test dataset is ",test_dataset.shape[0]," lines and ",test_dataset.shape[1], "columns") 
    print(test_dataset.individual.value_counts()) # Show number of picture by individual



    # -- 2) The images and labels ---------------------------------------------

    images_test, labels_test = [], []

    # Iterate over the rows of the newly created dataframe
    for (_, row) in tqdm(test_dataset.iterrows()):
        # Read the image from the disk, resize it, and scale the pixel
        # values to [0,1] range
        image = plt.imread(row['path'])
        images_test.append(image)

        # Parse the name of the person
        print(row['path'])
        labels_test.append(row['path'])

    images_test = np.array(images_test)
    labels_test = np.array(labels_test)
    print(images_test.shape, labels_test.shape)

    # Here the informations
    print("There are ", images_test.shape[0], " pictures, each picture has ", images_test.shape[1], " rows and ", images_test.shape[2], "columns. For each pixel, there are", images_test.shape[3]," values describing them (RGB red green blue)")
    print("Here the labels of the test dataset: ",labels_test)
    
    
    # ======= III/ PREDICT THE NEAREST NEIGHBOR FOR EACH PICTURE ==============
    
    # We take the dataframe for re-id
    reid_df = evaluate_all_return_csv(model_saved, images_test, labels_test,test_dataset,EVAL_DISTANCE)
    print(reid_df)
    
    
    # ======= IV/ PREPARE THE VECTOR WITH THE SUB GROUPS ======================
    
    # -- 1) The pictures alone ------------------------------------------------

    # Here all the pictures with dist > treshold which are alone in the folder
    m_alone = reid_df[reid_df.distance > TRESHOLD].pic_name.tolist()
    print(m_alone)
    
    # we separate them into several sub-vectors for later
    if len(m_alone) >= 1:
        alone = [[m_alone[0]]]
        for i in range(1,len(m_alone)):
            print(i)
            alone.append([m_alone[i]])
        print(alone)
    
    # -- 2) The pictures in group ---------------------------------------------
    
    # we need to put in group the others
    m_group = reid_df[reid_df.distance <= TRESHOLD]

    # We create a vector with sub-vector of each association
    vec_near = m_group.pic_near.tolist()
    vec_pic = m_group.pic_name.tolist()
    vec_m = []
    for i in range(len(vec_near )):
        vec_m.append([vec_near[i],vec_pic[i]])
    print(vec_m)
    
    # We merge the sub-vectors with common values
    out = put_together_common(vec_m)
    
    # And convert the list into a vector
    grouped = []
    for i in range(len(out)):
        grouped = grouped + [list(out[i])]
    print(grouped)
    
    
    # -- 3) And finally all the pictures in a vector with sub-vectors ---------
    
    if len(m_alone) >= 1:
        all_lynx = alone + grouped
    
    
    
    # ======= V/ OBTAIN A DATAFRAME WITH THE INDIVIDUAL AND THEIR NUMBER BY GROUP ======
    
    # -- 1) The dataframe individual / folder predicted -----------------------
    
    
    # We create the dataframe linking picture and folder
    df_folder_file = link_pic_folder(dir_pic_SCR)
    
    
    # We create the dataframe of the folders linked
    lynx = []
    for i in range(len(all_lynx)):
        pred_folder = "lynx" + str(i)
        for j in range(len(all_lynx[i])):
            individual = list(set(df_folder_file.folder[df_folder_file.file == all_lynx[i][j]]))[0]
            lynx.append((individual, pred_folder))
    print(lynx)
    df_re_id = pd.DataFrame(lynx, columns=['individual', 'pred_folder'])
    print(df_re_id)
    
    
    # -- 2) We obtain a count dataframe ---------------------------------------
    
    # Here the list of the folder :
    list_ind_folder = np.unique(df_re_id.pred_folder).tolist()
    
    
    # We initialize :
    # We have the count of each individual in this file
    ind_folder_all = df_re_id.individual[df_re_id.pred_folder==list_ind_folder[0]].value_counts()
    # We convert it into a dataframe
    ind_folder_all = pd.DataFrame({'ind':ind_folder_all.index, 'nb':ind_folder_all.values})
    # And add the folder name
    ind_folder_all = ind_folder_all.assign(folder = list_ind_folder[0])
        
    # And then do the same for the others :
    for i in range(1,len(list_ind_folder)):
        
        # We have the count of each individual in this file
        ind_folder = df_re_id.individual[df_re_id.pred_folder==list_ind_folder[i]].value_counts()
        # We convert it into a dataframe
        ind_folder = pd.DataFrame({'ind':ind_folder.index, 'nb':ind_folder.values})
        # And add the folder name
        ind_folder = ind_folder.assign(folder = list_ind_folder[i])
        
        # And add these lines to the main dataframe
        ind_folder_all = pd.concat([ind_folder_all,ind_folder])
        
    # And we finally reset the index
    ind_folder_all = ind_folder_all.reset_index(drop=True) 
    
    
    
    # -- 3) We assign names to folders ----------------------------------------
    
    # List of the real id of the individuals
    list_ind_id = df_re_id.individual.unique()
    print(list_ind_id)
    
    # We sort the dataframe by the number of individual pictures grouped
    df = ind_folder_all
    print(df)
    df = df.sort_values(by=['nb'], ascending=False)
    print(df)
    df = df.reset_index(drop=True)
    
    # Here an example of what we can obtain as dataframe 'df':
    #          ind  nb folder
    # 0  OCS_Minus   6  lynx3
    # 1  OCS_Letty   5  lynx3
    # 2  OCS_Arcos   1  lynx1
    # 3  OCS_Arcos   1  lynx2
    # 4  OCS_Letty   1  lynx4
    
    while len(df)>0:
        
        print(" ====== folder is ",df.folder[0]," ====== ")
        print(" ====== ind is ",df.ind[0]," ====== ")
        
        # And so this folder name (lynx1, or lynx2, ...)
        name_assigned = df.folder[0]
        print("name_assigned")
        print(name_assigned)
        
        # And the lynx name (OCS_Minus, ...)
        new_name = df.ind[0]
        print("new name")
        print(new_name)
        
        # We find the lines of this folder name (lynx1, or lynx2, ...)
        idx_folder = np.where((df_re_id.pred_folder == name_assigned) == True)
        print("idx_folder")
        print(idx_folder)
        
        # We rename this folder (lynx3 = OCS_Minus, ...)
        df_re_id.pred_folder.iloc[idx_folder] = new_name
        
        # We delete the lines already assigned (folder = lynx1 and ind = OCS_Minus here)
        print("df before deleted")
        print(df)
        df = df[df.folder != name_assigned]
        df = df[df.ind != new_name]
        df = df.reset_index(drop=True)
        print("df")
        print(df)
        
        
        
    # -- 4) We obtain a 2nd count dataframe with names assigned ---------------
    
    # Here the list of the folder :
    list_ind_folder = np.unique(df_re_id.pred_folder).tolist()
    
    
    # We initialize :
    # We have the count of each individual in this file
    ind_folder_all = df_re_id.individual[df_re_id.pred_folder==list_ind_folder[0]].value_counts()
    # We convert it into a dataframe
    ind_folder_all = pd.DataFrame({'ind':ind_folder_all.index, 'nb':ind_folder_all.values})
    # And add the folder name
    ind_folder_all = ind_folder_all.assign(folder = list_ind_folder[0])
        
    # And then do the same for the others :
    for i in range(1,len(list_ind_folder)):
        
        # We have the count of each individual in this file
        ind_folder = df_re_id.individual[df_re_id.pred_folder==list_ind_folder[i]].value_counts()
        # We convert it into a dataframe
        ind_folder = pd.DataFrame({'ind':ind_folder.index, 'nb':ind_folder.values})
        # And add the folder name
        ind_folder = ind_folder.assign(folder = list_ind_folder[i])
        
        # And add these lines to the main dataframe
        ind_folder_all = pd.concat([ind_folder_all,ind_folder])
        
    # And we finally reset the index
    ind_folder_all = ind_folder_all.reset_index(drop=True) 
    
    
    
        
    # -- 5) We see if well or bad classed -------------------------------------
    
    # We say if it's well classed in the group or not
    ind_folder_all['well_classed'] = "no"
    ind_folder_all['well_classed'][ind_folder_all.ind==ind_folder_all.folder] = "yes"
    
    # We show when it create a new individual being alone
    ind_folder_all['alone_and_new'] = "no"
    for i in range(len(ind_folder_all)):
        if ((ind_folder_all.folder == ind_folder_all.folder[i]).sum() == 1) and (ind_folder_all.nb[i] == 1):
            ind_folder_all.alone_and_new[i] = "yes"
    
    # We save it
    path_save = "D:/deep-learning_re-id_gimenez/7_Examine-distribution/result_distributionTRAIN.csv"
    ind_folder_all.to_csv(path_save, sep=';',index=False)


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
TRESHOLD = 0.4


# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# FOLDER SPLIT : Folder with split images which will be used for the deep-learning
dir_split = "D:/deep-learning_re-id_gimenez/1_Pre-processing_min_pic_15/dataset_split_260"
print("The directory of the splitted pictures is ",dir_split,"\n")

# FOLDER PICTURES
folder_test = "train"
dir_pic_SCR = os.path.join(dir_split,folder_test)
print("dir_pic_SCR is ",dir_pic_SCR)

# BEST MODEL DIRECTORY : Where the best model is
dir_best_model = "D:/deep-learning_re-id_gimenez/2_Model-construction_batchcompo/result_model/260size_squared-L2_NCB10_NIC3_EfficientNet"

# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

#disable chained assignments
pd.options.mode.chained_assignment = None 

distribution(my_dir_split=dir_split,my_dir_best_model=dir_best_model,my_folder_test=folder_test)





