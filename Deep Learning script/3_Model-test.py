# -*- coding: utf-8 -*-
"""
PROJET DEEP LEARNING LYNX - Internship M2 2023

Author : Marie Joigneau
Supervisor : Olivier Gimenez

Problematic : Re-identify individual and estimate population density from eurasian lynx population

Before : 
- We have prepared the pictures for the model
- We have created different models depending of different parameters

This part : Here we will test theses different models to choose the best model
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
# ======================== FUNCTIONS ==========================================
# =============================================================================
# =============================================================================

"""
Title: predict_class_test

Description: return the dataframes of re-identification of the nearest neighbors for each picture

Usage: predict_class_df_treshold(model, images, train_labels, labels,df,my_eval_distance,ind_new_id)

Argument: 
-	model (the deep learning model)
-   images (the pictures to analyze)
-   labels (the identities of the pictures)
-   df (the dataframe with all the informations on the pictures)
-   my_eval_distance (metric chosen of the k-neighbors)

Details: 
Return the dataframes of re-identification of the nearest neighbors for each picture

Values: 
-   reid_dataframe : dataframe for the individuals to re-identify
"""

def predict_class_test(model, images, labels,df,my_eval_distance):

    # Results of the model prediction : extract the 128 dimension vectors (x number of picture) of the embeddings from the test images
    test_features = model.predict(images) 

    # Train a nearest neighbors model with the extracted embeddings. Here we are limiting it to 11 neighbors only
    neighbors = NearestNeighbors(n_neighbors=11, 
        algorithm='brute',
        metric=my_eval_distance).fit(test_features) 
    
    # We initiate the re-identification dataframe
    reid = []

    # When you use enumerate(), the function gives you back two loop variables:
    # - The count of the current iteration
    # - The value of the item at the current iteration
    
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
        print("The similar labels (k voisins) are :")
        print(similar_labels)
        print("The labels of the individual is ",labels[indices[0][0]])
        
        # i_top : permit to know the n nearest individual which is the same as the picture
        # if i_top = 11, we consider that it's too far away and pu a value by default
        i_top = 0
        
        # For each individual :
        for id2, similar_label in enumerate(similar_labels):
            
            # In the k neighbors of the individual n°id, we find the k-th neighbors that verify the initial label
            if similar_label == labels[id]:
                # In the result :
                # - id = the id of the test individual
                # - id2+1 = we put +1 because id2=0 is the test individual
                # - distances[0][id2+1] = We save the distance of the closest neighbors 
                # - id2+1 : the k-th closest neighbors that verify the identity of the test individual
                result = (id, distances[0][id2+1], id2+1,os.path.basename(df.path.iloc[id]))
                print("(RE-ID) The distances of the closest right label is ",distances[0][id2+1])
                break
            
            # if we don't manage to find an image of the same individual in the top 11
            if i_top == (len(similar_labels)-1): 
                print("i_top = 11")
                result = (id, 100, 11, "none")
                
            i_top = i_top +1
            print("i_top is ",i_top)
            
        # The result goes in the reid_df
        reid.append(result)
    
    reid_dataframe = pd.DataFrame(reid, columns=("id", "distance", "top","pic_name"))
    
    print("The Re-ID dataframe is : ")
    print(reid_dataframe)
    
    return(reid_dataframe)


"""
Title: eval_metric

Description: calcul the different metrics CMC@k and mAP@k to test the models

Usage: eval_metric(df)

Argument: 
-	df (dataframe with the results of re-identification)

Details: 
    
Calcul the different metrics CMC@k (k=1 and 5) and mAP@k (k = 1 and 5) to test the models

Accuracy top-k :
        A(cck) = 1 if top-k embeddings contain the query identity
        A(cck) = 0 otherwise
        
CMC@k : Cumulative Matching Characteristics top-k
Defined as the mean of the Accuracy top-k. This metric only shows if the correct individual is in the top-k without considering the order.
CMC@k = 1/N * SUM( A(cck) )

Our mAP@k is a variant of the one used for the Kaggle challenge "Humpback Whale Identification"
mAP@k = 1/N * SUM( SUM ( P(j) * rel(j) ) )
P(j) is the precision at top-j
P(j) = TP / (TP + FP) = TRUE POSITIVE / (TRUE POSITIVE + FALSE POSITIVE) 
rel(j) = 1 if the individual is the same + 1st match
rel(j) = 0 or 1 


Values: 
-   cmc1 : CMC@1
-   cmc5 : CMC@5
-   map1 : mAP@1
-   map5 : mAP@5
"""


def eval_metric(df):
    
    # The metrics we will be using :
    # CMC@1 CMC@5 
    # And mAP@1 mAP@5
    
    # ======= I/ CUMULATIVE MATCHING CHARACTERISTICS (CMC@k) ==================
    
    N = len(df)
    print("The number of individual is is ",N)
    
    # Corrects individuals in the top 1?
    cmc1 = (1/N)*len(df[df.top==1])
    cmc1 = round(cmc1,9)
    print("CMC@1 is ",cmc1)
    
    # Corrects individuals in the top 5?
    cmc5 = (1/N)*len(df[df.top<=5])
    cmc5 = round(cmc5,9)
    print("CMC@5 is ",cmc5)
    
    
    # ======= II/ MEAN AVERAGE PRECISION AT K (mAP@k) =========================
    
    # Top 1 is the correct one?
    # mAP@1 = CMC@1
    map1 = (1/N)*len(df[df.top==1])
    map1 = round(map1,9)
    print("mAP@1 is ",map1)
    
    # We take into account the position of the 1st correct match (in the top 5) :
    # 'np.mean' replace the '(1/N)*'
    # We are in the top 5. If the result is > 5, the map5 is 0.
    map5 = []
    for i in range(len(df)):
        if df.top.iloc[i] >5 :
            map5_i = 0
        else:
            map5_i = 1/df.top.iloc[i]
        map5 = map5 + [map5_i]
    print("mAP@5 before mean is ",map5)
    map5 = np.mean(map5)
    map5 = round(map5,9)
    print("mAP@5 is ",map5)
    
    return(cmc1,cmc5,map1,map5)


# =============================================================================
# ===  MAIN FUNCTION ==========================================================
# =============================================================================

"""
Title: modeltest

Description: test the different models

Usage: modeltest(my_dir_split_list,my_dir_model,my_eval_distance_list,my_triplet_choice_list,my_num_classes_per_batch_list,my_num_images_per_class_list,my_dir_model_test,my_size_list,my_fold_test)

Argument: 
-   my_dir_split_list : list of all the directories of the splitted pictures to be used for the different models
-   my_dir_model : directory of the different models to test
-   my_eval_distance_list : list of the k neighbors metrics to test
-   my_triplet_choice_list : the list of the metrics of the triplet loss to test
-   my_num_classes_per_batch_list : list of the number of class by batch
-   my_num_images_per_class_list : list of the number of image by class by batch
-   my_size_list : list of all the sizes to test
-   my_fold_test : folder where the pictures to test are

Details: 
    
First, we test different models depending on : 
-   the size of the pictures
-   the premodel chosen for the transfer learning
-   the metric of the triplet loss
-   the metric of the k neighbors
For this part, NUM_CLASSES_PER_BATCH and NUM_IMAGES_PER_CLASS must be of len 1 (like = [2] and [10])

Then for the best parameters above, we test different models depending on : 
-   the size of the pictures
-   the premodel chosen for the transfer learning
-   the number of class by batch
-   the number of image by class by batch
For this part, ALL_SIZE, TRIPLET_CHOICE, EVAL_DISTANCE and PREMODEL_CHOSEN  must be of len 1
( like = [224], = ["squared-L2"], = ["cosine"] and = ["EfficientNet"]

Values: return a csv with the parameters and the evaluation metrics CMC@k and mAP@k
"""


def modeltest(my_dir_split_list,my_dir_model,my_eval_distance_list,my_triplet_choice_list,my_num_classes_per_batch_list,my_num_images_per_class_list,my_dir_model_test,my_size_list,my_fold_test,my_embedding_size_list,my_first_dense_size_list):
    
    result = []
 
    nb_dir_split = 0
    
    print("The list of my different size to test is ",my_size_list)
    print("The list of my splitted pictures directories to test is ",my_dir_split_list)
    
    # ======= I/ CHOICE OF THE FOLDER TO TEST (DEPENDING OF THE SIZE) =========
    
    for my_size in my_size_list:
        
        # We take the folder with the pictures of the size chosen
        my_dir_split = my_dir_split_list[nb_dir_split]
        nb_dir_split = nb_dir_split + 1
        print("The directory of my splitted pictures is ",my_dir_split)
        print("The size of my pictures to test is ",my_size)
                
        # ======= II/ PREPARE THE TEST DATASET ================================
        # -- 1) The dataframe ---------------------------------------------------------
        
        test_dataset = []
        
        # Iterate over the folders (individuals)
        for indiv in os.listdir(os.path.join(my_dir_split, my_fold_test)):
            for picturename in os.listdir(os.path.join(my_dir_split, my_fold_test, indiv)):
                test_dataset.append({'individual':indiv, 'path': os.path.join(my_dir_split, my_fold_test, indiv, picturename)})
        
        # Prepare a dataframe and preview it
        test_dataset = pd.DataFrame(test_dataset)
        test_dataset.head(10)
        
        
        #Let's analyse that train dataset
        print("The shape of the test dataset is ",test_dataset.shape[0]," lines and ",test_dataset.shape[1], "columns") 
        print("The minimal picture number by class is : ",np.min(test_dataset["individual"].value_counts())) # Minimal number of picture for the classes
        
        nb_individual = len(os.listdir(os.path.join(my_dir_split, my_fold_test)))
        print("There are ",nb_individual," individuals in the test dataset")
        
        ind_test_unique=list(test_dataset["individual"])
        ind_test_unique=set(ind_test_unique)
        ind_test_unique=list(ind_test_unique)
        print("Here the individuals of the test dataset : ",ind_test_unique)
        
        # Show number of picture by individual
        print(test_dataset.individual.value_counts()) 
        
        
        # -- 2) The test images and labels --------------------------------------------
        
        images_test, labels_test = [], []
        
        # Iterate over the rows of the newly created dataframe
        for (_, row) in tqdm(test_dataset.iterrows()):
            image = plt.imread(row['path'])
            images_test.append(image)
            labels_test.append(row['individual'])
        
        images_test = np.array(images_test)
        labels_test = np.array(labels_test)
        print(images_test.shape, labels_test.shape)
        
        # Here the informations
        print("There are ", images_test.shape[0], " pictures, each picture has ", images_test.shape[1], " rows and ", images_test.shape[2], "columns. For each pixel, there are", images_test.shape[3]," values describing them (RGB red green blue)")
        print("Here the labels of the test dataset: ",labels_test)
        
        
        # ======= III/ EVALUATE BY K-NEIGHBORS ================================
        # -- 1) Call the model --------------------------------------------------------
        
        # For each triplet distance choice :
        for my_triplet_choice in my_triplet_choice_list:
            
            print("my_triplet_choice is ",my_triplet_choice)
            print("my size is ",my_size)
            
            # When we want to compare batch compositions
            for i in range(len(my_num_classes_per_batch_list)):
                
                my_num_classes_per_batch = my_num_classes_per_batch_list[i]
                my_num_images_per_class = my_num_images_per_class_list[i]
                
                for k in range(len(my_embedding_size_list)):
                    
                    my_embedding_size = my_embedding_size_list[k]
                    my_first_dense_size = my_first_dense_size_list[k]
                    print("The embedding size is ",my_embedding_size)
                    print("The first dense size is ",my_first_dense_size)
                
                    # Here the directory to save the model :
                    # - either we want different size layers
                    # - or we want to choose different premodels / size ect
                    # - or the compositions of the batchs
                    if len(my_embedding_size_list) > 1 :
                        # embedding : size of the second dense
                        # firstdense : size of the first dense
                        save_num_classes_images_triplet_choice = str(my_size) + "size_"  + str(my_triplet_choice) + "_firstdense" + str(my_first_dense_size) + "_embedding" + str(my_embedding_size) + "_" + str(PREMODEL_CHOSEN)
                    if len(my_num_classes_per_batch_list) > 1 :
                        # NCB : number of class by batch
                        # NIC : number of image by class
                        save_num_classes_images_triplet_choice = str(my_size) + "size_"  + str(my_triplet_choice) + "_NCB" + str(my_num_classes_per_batch) + "_NIC" + str(my_num_images_per_class) + "_" + str(PREMODEL_CHOSEN)
                    if len(my_num_classes_per_batch_list) ==1 and len(my_embedding_size_list)==1:
                        save_num_classes_images_triplet_choice = str(my_size) + "size_" + str(my_triplet_choice) + "_" + str(PREMODEL_CHOSEN)
                    my_dir_model_triplet = os.path.join(my_dir_model,save_num_classes_images_triplet_choice)
                    
                    print("The directory of my model is")
                    print(my_dir_model_triplet)
                    
                    # We can load the model after if needed
                    model_saved = tf.keras.models.load_model(my_dir_model_triplet)
                    
                    # -- 2) With only the dataframe -----------------------------------------------
                    
                    print("The true labels of the test dataset are ",labels_test,"\n")
                    
                    # For each evaluation distance choice :
                    for my_eval_distance in my_eval_distance_list:
                        
                        print("----- The metric of the Triplet Loss is ",my_triplet_choice,"-----------")
                        print("The size of my picture is ",my_size)
                        print("The metric of the k neighbors is ",my_eval_distance)
                        print("My dataset test is ",my_fold_test)
                        print("There are ",my_num_classes_per_batch," classes by batch")
                        print("There are ",my_num_images_per_class," images by class in a batch")
                        print("The embedding size is ",my_embedding_size)
                        print("The first dense size is ",my_first_dense_size)
                        
                        reid_df = predict_class_test(model_saved, images_test, labels_test,test_dataset,my_eval_distance)
                        
                        
                        # -- 3) With the metrics applied to the dataframe -----------------------------
    
                        cmc1,cmc5,map1,map5 = eval_metric(reid_df)
                        
                        
                        # Here the directory to save the model :
                        # - either we want different size layers
                        # - or we want to choose different premodels / size ect
                        # - or the compositions of the batchs
                        if len(my_embedding_size_list) > 1 :
                            # embedding : size of the second dense
                            # firstdense : size of the first dense
                            result_metrics = (my_size,PREMODEL_CHOSEN,my_first_dense_size,my_embedding_size,cmc1,cmc5,map1,map5)
                        if len(my_num_classes_per_batch_list) > 1 :
                            # NCB : number of class by batch
                            # NIC : number of image by class
                            result_metrics = (my_size,PREMODEL_CHOSEN,my_num_classes_per_batch,my_num_images_per_class,cmc1,cmc5,map1,map5)
                        if len(my_num_classes_per_batch_list) ==1 and len(my_embedding_size_list)==1:
                            result_metrics = (my_size,PREMODEL_CHOSEN,my_triplet_choice,my_eval_distance,cmc1,cmc5,map1,map5)
                        result.append(result_metrics)
                        
                        print("The result is")
                        print(result)        
            
            
    # ======= IV/ SAVE THE RESULTS ============================================
    
    if len(my_embedding_size_list) > 1 :
        # embedding : size of the second dense
        # firstdense : size of the first dense
        result_df = pd.DataFrame(result, columns=("size","premodel","size_first","size_second","CMC@1", "CMC@5", "mAP@1","mAP@5"))
    if len(my_num_classes_per_batch_list) > 1 :
        # NCB : number of class by batch
        # NIC : number of image by class
        result_df = pd.DataFrame(result, columns=("size","premodel","nb_class","nb_image_class","CMC@1", "CMC@5", "mAP@1","mAP@5"))
    if len(my_num_classes_per_batch_list) ==1 and len(my_embedding_size_list)==1:
        result_df = pd.DataFrame(result, columns=("size","premodel","triplet_dist","eval_dist","CMC@1", "CMC@5", "mAP@1","mAP@5"))

    print("The dataframe with all the result is :")
    print(result_df)
        
    name_csv = "results_all_test.csv"
        
    path_save = os.path.join(my_dir_model_test,name_csv)
    result_df.to_csv(path_save, sep=';')
    

# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

# SIZE : size of the pictures
# For testing one model :
ALL_SIZE = [260]
# For testing different models :
#ALL_SIZE = [224, 240, 260, 300, 380, 456, 528, 600]

# NUM_CLASSES_PER_BATCH : number of class by batch
# NUM_IMAGES_PER_CLASS : number of image by class by batch
# If you want to choose a single model :
NUM_CLASSES_PER_BATCH = [10]
NUM_IMAGES_PER_CLASS = [3]
# If you want to test different models
#NUM_CLASSES_PER_BATCH = [3,10,15,6,3,20,10,30,15]
#NUM_IMAGES_PER_CLASS = [10,3,2,10,20,3,6,2,4]

# TRIPLET_CHOICE : choice of the metric of the triplet loss function
# - If you want to choose a single model :
TRIPLET_CHOICE = ["squared-L2"]  
# - If you want to test different model by modifying triplet distance :
#TRIPLET_CHOICE = ["L2","squared-L2","angular"]

# Here we are extracting128-dimensional embedding vectors.
# If you want to choose a single model :
EMBEDDING_SIZE = [128]
FIRST_DENSE = [2048]
# If you want to test different models
#EMBEDDING_SIZE = [64,64,64,128,128,128]
#FIRST_DENSE = [2048,1024,512,2048,1024,512]

# EVAL_DISTANCE : metric of the evaluation distance in the k neighbors
#EVAL_DISTANCE = ["cosine"]
# If you want to search the best model by modifying:
EVAL_DISTANCE = ["cosine"]

# PREMODEL_CHOSEN
# - Here we put the different pre-models
# /!\ For ResNet152 et DenseNet201 we need to put ALL_SIZE = 224
# /!\ For the different EfficientNet model, the ALL_SIZE depend of the model
# PREMODEL = ["EfficientNet","ResNet152","DenseNet201"]
PREMODEL_CHOSEN = "EfficientNet"

# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# MAIN DIRECTORY : Where we have everything except the code
#dir_main = 'D:/deep-learning_re-id_gimenez'
dir_main = "D:/deep-learning_re-id_gimenez/OLIVIER/A envoyer OFB jaguar"
# dir_main_data = "/home/sdb1"
# dir_main_save = "/home/data/marie"
dir_main_data = dir_main
dir_main_save = dir_main
#print("The main directory is ",dir_main,"\n")

# DIR SPLIT : directories with split images which will be used for the deep-learning
# We have different sizes depending of the different models to test
dir_split_all = []
for size_chosen in ALL_SIZE:
    folder_animal_split = 'dataset_split_' + str(size_chosen)
    dir_split = os.path.join(dir_main_data,"1_Pre-processing_jaguar", folder_animal_split)
    dir_split_all = dir_split_all + [dir_split]
print("The directory of the splitted pictures is ",dir_split_all,"\n")

# FOLDER TEST : folder with all the pictures to test
fold_test = "test_1-photo-supprimee"

# FOLDER MODEL CONSTRUCTION : where the model, its history and mapping will be saved
dir_construction = os.path.join(dir_main_save,"2_Model-construction_jaguar")
# The model directory :
dir_model = os.path.join(dir_construction,"result_model")

# FOLDER RESULT METRICS : where the results of the tests will be saved
dir_model_test = os.path.join(dir_main_save,"3_Model-test")

# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

modeltest(my_dir_split_list=dir_split_all,my_dir_model=dir_model,my_eval_distance_list=EVAL_DISTANCE,my_triplet_choice_list=TRIPLET_CHOICE,my_num_classes_per_batch_list=NUM_CLASSES_PER_BATCH,my_num_images_per_class_list=NUM_IMAGES_PER_CLASS,my_dir_model_test=dir_model_test,my_size_list=ALL_SIZE,my_fold_test=fold_test,my_embedding_size_list=EMBEDDING_SIZE,my_first_dense_size_list=FIRST_DENSE)


