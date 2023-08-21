# -*- coding: utf-8 -*-
"""
PROJET DEEP LEARNING LYNX - Internship M2 2023

Author : Marie Joigneau
Supervisor : Olivier Gimenez

Problematic : Re-identify individual and estimate population density from eurasian lynx population

This part : Here we will prepare the pictures to construct and test deep learning models
"""

# =============================================================================
# =============================================================================
# ======================== PACKAGES ===========================================
# =============================================================================
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json5
import os
import matplotlib.pyplot as plt
import shutil
import imageio.v2 as imageio
import imgaug.augmenters as iaa
from PIL import ImageFile, Image



# =============================================================================
# =============================================================================
# ======================== FUNCTIONS ==========================================
# =============================================================================
# =============================================================================


"""
Title: folder_description

Description: change the minimal number of picture by individual in a dataset

Usage: folder_description(my_dir_pop)

Argument: 
-	my_dir_pop (directory of all the pictures classified in folders in the population)

Details: 
Describe the repartition of pictures in the population :
-   max_picture_all (maximum number of picture by individual)
-   mean_picture_all (mean number of picture by individual)
-   min_picture_all (minimal number of picture by individual)
Then return the list of individuals with 1 picture

Values: 
-   ind_1_picture : list of individuals with only 1 picture
"""

def folder_description(my_dir_pop):
    
    # Individual list :
    list_ind = os.listdir(my_dir_pop) 
    print("Here the list of all the individuals : ",list_ind)
    
    # Initialisation of the 1-file animal vector
    ind_1_picture = [] 
    
    # Initialisation of the vector containing all the picture count for each individual
    nb_picture = []

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
        
        # We add the picture count for this individual
        nb_picture = nb_picture + [initial_count]
    
    # Then we have a description of the folder :   
        
    max_picture_all = np.max(nb_picture)
    print("Maximum number of picture by individual")
    print(max_picture_all)
    
    mean_picture_all = np.mean(nb_picture)
    print("Mean number of picture by individual")
    print(mean_picture_all)
    
    min_picture_all = np.min(nb_picture)
    print("Minimum number of picture by individual")
    print(min_picture_all)
    
    # Ouput : the vector of the animal names with 1 picture
    return(ind_1_picture)



"""
Title: delete_pic_1_folder

Description: delete folders with only 1 picture

Usage: delete_pic_1_folder(my_dir)

Argument: 
-	my_dir (directory of all the pictures classified in folders in the population)

Details: 
Delete folders with only 1 picture

Values: /
"""


def delete_pic_1_folder(my_dir):
    
    # We take the list of individuals with 1 picture
    list_1_pic = folder_description(my_dir)
    print("Here the list of all the folders with only 1 file")
    print(list_1_pic)
    
    # Then we delete the folders / individuals with 1 pictures 
    for pic in list_1_pic:
        my_dir_pic = os.path.join(my_dir,pic)
        shutil.rmtree(my_dir_pic)




"""
Title: from_json_to_df

Description: takes the output of Megadetector which focus the animal in the picture
and converts it into a dataframe

Usage: from_json_to_df(my_megadetect_path, my_dir_main, my_dir_bbox_pb,treshold_bbox)

Argument: 
-	my_megadetect_path : directory where the Megadetector file is (JSON file)
-   my_dir_main : main directory of the pictures
-   my_dir_bbox_pb : directory of the folder with bounding boxes examples of problems
-   treshold_bbox : minimal value of confidence to automatically delete some bounding boxes
proposed

Details: 
    
Takes the output of Megadetector which focus the animal in the picture
and converts it into a dataframe. Permits to choose between the different bounding
boxes of Megadetector for a picture or to automatically take the best one.

Here the explaination to obtain a .json file with the Megadetector :
  -> https://github.com/microsoft/CameraTraps/blob/main/megadetector.md
The goal of Megadetector in our case is to have the coordinates of the bbox around the animal

https://gitlab.com/ecostat/imaginecology/-/blob/master/projects/detectionWithMegaDetector/README.md
The dataframe we want to obtain contain all the informations about the bbox, it was extracted from the .json file of the megadetector :
file (1)
max_detection_conf (2)
detections
  -> category (3)  : 0 for "empty", 1 for "animal", 2 for "person", 3 for "group" and 4 for "vehicle"
  -> conf (4)  
  -> bbox  
      => xmin (5)  
      => xmax (6)  
      => width (7)  
      => height (8)  
There will be 8 columns as indicated. The bounding box coordinates [xmin, xmax, width, height] are all in relative value to the image dimension.

Values: 
-   df_json : dataframe with values of the bounding box around the animal
"""


def from_json_to_df (my_megadetect_path, my_dir_main, my_dir_bbox_pb,treshold_bbox):
    
    
    
    # ======= I/ OBTAINING A FIRST DATAFRAME ==================================
    # -- 1) A dataframe with 1 line -------------------------------------------
    print("======= I/ OBTAINING A FIRST DATAFRAME ==================================")
    
    
    # https://www.geeksforgeeks.org/read-json-file-using-python/ 

    # We open the JSON file
    f = open(os.path.join(my_dir_main, my_megadetect_path))

    # We returns JSON object as a dictionary
    data = json5.load(f)

    # We initialise by extracting all the variables from the .json file. 
    df_json = pd.DataFrame(data['images'][0]['detections'])
    df_json["file"] = data['images'][0]['file']
    df_json['max_detection_conf'] = data['images'][0]['max_detection_conf']
    df_json["xmin"] = df_json["bbox"][0][0]
    df_json["xmax"] = df_json["bbox"][0][1]
    df_json["width"] = df_json["bbox"][0][2]
    df_json["height"] = df_json["bbox"][0][3]
    del df_json['bbox'] # We don't need anymore the bbox column as we have divided it into the 4 precedent columns
    print(df_json.head())
    print("\n The shape of the dataframe is ",df_json.shape[0]," line and ",df_json.shape[1], "columns") # 1 line, 8 columns !
    
    # -- 2) A dataframe with all the lines ------------------------------------

    # For all the others pictures we extract all the variables from the .json file. 
    for i in range(1,(len(data['images']))) :
        
        # If we don't find any bbox, we don't put it in the dataframe
        if data['images'][i]['max_detection_conf'] != 0.0 : 
            
            # df : dataframe for 1 picture
            df = pd.DataFrame(data['images'][i]['detections']) # We divid the detection variable into its 3 components (category, conf and bbox) thanks to a dataframe df 
            df["file"] = data['images'][i]['file'] # We add the file variable
            df['max_detection_conf'] = data['images'][i]['max_detection_conf'] # We add the max_detection_conf variable
            
            # df_bbox : dataframe of the bbox values for 1 picture
            df_bbox = pd.DataFrame(list(df["bbox"])) # We divid the bbox variable into its 4 components (xmin,xmax,width,height) thanks to a dataframe 'df_bbox'
            df_bbox.columns = ["xmin","xmax","width","height"] # We rename the 4 columns
    
            # We concate the 2 dataframes df and df_bbox
            df = pd.concat([df, df_bbox], axis=1, join='inner') # Merge of 2 dataframes by adding columns
    
            # df_json : dataframe for all the i first pictures
            # - we concate the dataframe df for 1 picture to the dataframe df_json of all the i pictures
            df_json = pd.concat([df_json,df]) # Merge of 2 dataframes by adding rows
           
        else :
            print("no bbox")
            
    # And we delete the last column not used (because already divided into 4 other columns)
    del df_json['bbox']
    print(df_json.head())
    print("\n The shape of the dataframe is ",df_json.shape[0]," lines and ",df_json.shape[1], "columns")
    print("A problem is noticed: there are ",len(data['images'])," pictures, more than the ",df_json.shape[0],"bbox")
    print("In our study, we want to focus on 1 individual in 1 picture, so just have 1 bbox")
    
    
    
    
    # ======= II/ FULL FILTER OR NOT ==========================================
    
    auto_or_manual = input("do you want to filter automatically? (yes/no) ")
    
    # -- 1) We filter automatically -------------------------------------------
    
    if auto_or_manual == "yes":
        df_json.drop_duplicates(subset ="file", keep = 'last', inplace=True)
        
    
    # -- 2) We filter manually ------------------------------------------------
    
    else:
        
        # -- 2.A) 1st automatic filter ----------------------------------------
    
        # We filter the dataframe to keep a conf > treshold + category == 1 (1 = animal)
        df_step1 = df_json[df_json.conf>treshold_bbox]
        df_step1 = df_step1[df_step1['category']==str(1)]
    
        # And we reset the index
        df_step1 = df_step1.reset_index(drop=True)
    
        print(df_step1.head())
        print("The shape of the dataframe is ",df_step1.shape[0]," lines and ",df_step1.shape[1], "columns")
        print("We have ",df_step1.shape[0], "lines (bbox) in the dataframe but ",len(data['images'])," pictures")
        print("We need to check bbox problems with ",df_step1.shape[0]-len(data['images'])," images")
        
        
        # -- 2.B) Creating a sub-dataframe with all the bbox pictures with problems --------------

        # We found the pictures with several bbox
        inds = list(df_step1["file"])
        duplicates = [ind for ind in inds if inds.count(ind) > 1]
        unique_duplicates = list(set(duplicates))
        print(unique_duplicates)
    
        # Then we subset the .json dataframe
        df_step2 = df_step1[df_step1.file.isin(unique_duplicates)]
        # And we reset the index
        df_step2 = df_step2.reset_index(drop=True)
        print(df_step2)
        
        # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python 
        # We get the unique values
        ind_step2_unique=list(df_step2["file"])
        ind_step2_unique=set(ind_step2_unique)
        ind_step2_unique=list(ind_step2_unique)
        print("Here the individuals of the step2 dataframe with a problem : ",ind_step2_unique)
        
        
        # -- 2.C) Creating a folder with all the bbox pictures with problems --------------
        
        # We create the folder with all the bbox pictures with problems
        os.makedirs(my_dir_bbox_pb)
    
        path_bbox_problem = []
    
        # Then we do bbox crop with it. We will crop all the pictures :
        for i in range(0,df_step2.shape[0]):
    
            # We find the picture :
            file_on_df = df_step2.file[i]
            print(file_on_df)
            dir_ind = os.path.join(dir_pop,file_on_df) # We go in the folder of the individual ind
            print(dir_ind)
            img = Image.open(dir_ind) 
            img = img.convert('RGB')
    
            # We find the dimensions of the picture
            width_img = img.size[0] 
            height_img = img.size[1] 
    
            # We put the coordinates for the cropping
            left = int(df_step2["xmin"][i]*width_img);left 
            bottom = int(df_step2["xmax"][i]*height_img);bottom 
            right = int((df_step2["width"][i]+df_step2["xmin"][i])*width_img);right
            top = int((df_step2["height"][i]+df_step2["xmax"][i])*height_img);top 
    
            # We crop the picture
            img_res = img.crop((left, bottom, right, top))
    
            # We resize the picture
            img_pixel = img_res.resize((SIZE,SIZE))
    
            ind = os.path.basename(file_on_df) # Give the individual name
            print(ind)
    
            # We save the picture
            dir_save = my_dir_bbox_pb + "/idx" + str(i) + "_" + ind # We go in the folder of the individual ind
            print(dir_save)
    
            path_bbox_problem = path_bbox_problem + [dir_save]
    
            img_pixel.save(dir_save)
        
        print("The pictures with problem are now in your folder ",my_dir_bbox_pb)
        print("Please look at them")
        following = input("Note the pictures you want to keep, and when you are ready, enter a letter")
        
        
        
        # -- 2.D) Deleting manually the problems -----------------------------------
        
        
        # Then we delete manually the remaining ones thanks to the dataframe
        
        # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python (get unique value)
        ind_step1_unique=list(df_step1["file"])
        ind_step1_unique=set(ind_step1_unique)
        ind_step1_unique=list(ind_step1_unique)
        print("Here the individuals of the json dataframe : ",ind_step1_unique)
    
        list_delete = []
        for i in range(0,len(ind_step1_unique)):
            df_subset = df_step1[df_step1["file"]==ind_step1_unique[i]]
            if (len(df_subset)>1):
                print("================")
                print("i is ",i," on ",len(ind_step1_unique))
                print("The individual is ",ind_step1_unique[i])
                print(df_subset)
                for i in range(0,(len(df_subset)-1)):
                    bbox_delet = int(input("Write the line of the bbox you don't want (in the order) : "))
                    list_delete = list_delete + [bbox_delet]
        
        print("Here the list of bbox we will delete:")
        print(list_delete)
    
    
    
        # -- 2.E) Having the final dataframe -----------------------------------
        
        # With the list, we delete the lines of the bbox we don't want
        df_json = df_step1.drop(df_step1.index [ list_delete ])
        print(df_json)
    
    
    
    # ======= III/ FINAL DATAFRAME ============================================
    
    # And we reset the index
    df_json = df_json.reset_index(drop=True)

    # Final and beautiful view :
    print(df_json.head())
    print("The shape of the dataframe is ",df_json.shape[0]," lines and ",df_json.shape[1], "columns")
    print(df_json)

    # Closing file
    f.close()
    
    print("\nWe have ",df_json.shape[0], "lines (bbox) in the dataframe as the ",len(data['images'])," pictures")
    
    return(df_json)




"""
Title: crop_resize

Description: crop and resize all the pictures in their folders

Usage: crop_resize (my_dir_crop_resize,my_dir_main,my_folder_raw_animal,my_df_json_final)

Argument: 
-	my_dir_crop_resize : directory where all the pictures are
-   my_dir_main : main directory
-   my_folder_raw_animal : folder with the pictures
-   my_df_json_final : dataframe with coordinates of bounding boxes of the pictures

Details: 
    
Takes the output of Megadetector which focus the animal in the picture

Values: /
"""



def crop_resize (my_dir_crop_resize,my_dir_main,my_folder_raw_animal,my_df_json_final):
    
    
    # ======= I/ CREATING AND COPYING THE FOLDERS =============================
    
    # We create the folder with all the crop resize pictures
    os.makedirs(my_dir_crop_resize)
    
    # We need first to copy all the folder names with the labels of the individuals to the new crop folder
    labels_ind = os.listdir(dir_pop) # We take the list of the folders names = list of the individual names
    print("Here the different labels ",labels_ind)
    for i in range(len(labels_ind)):
        newpath = my_dir_crop_resize + "/" + labels_ind[i]
        os.makedirs(newpath)
        
    
    # ======= II/ CROP AND RESIZE THE PICTURES ================================

    # We will crop all the pictures :
    for i in range(0,my_df_json_final.shape[0]):

        # We find the picture :
        file_on_df = my_df_json_final.file[i]
        print(file_on_df)
        dir_ind = os.path.join(my_dir_main, my_folder_raw_animal,file_on_df) # We go in the folder of the individual ind
        print(dir_ind)
        img = Image.open(dir_ind) 
        img = img.convert('RGB') # necessary if we have transparent pictures (https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg)

        # We find the dimensions of the picture
        width_img = img.size[0] 
        height_img = img.size[1] 

        # We put the coordinates for the cropping
        left = int(my_df_json_final["xmin"][i]*width_img);left 
        bottom = int(my_df_json_final["xmax"][i]*height_img);bottom 
        right = int((my_df_json_final["width"][i]+my_df_json_final["xmin"][i])*width_img);right
        top = int((my_df_json_final["height"][i]+my_df_json_final["xmax"][i])*height_img);top 


        # EXPLAINATION OF THE IMG.CROP() FUNCTION
        # https://www.askpython.com/python/examples/crop-an-image-in-python
        # top and left: These parameters represent the top left coordinates i.e (x,y) = (left, top).
        # bottom and right: These parameters represent the bottom right coordinates i.e. (x,y) = (right, bottom).
        # The area to be cropped is represented as follows:
        # left <= x < right
        # top <= y < bottom
        
        # We crop the picture
        img_res = img.crop((left, bottom, right, top))

        # We resize the picture
        img_pixel = img_res.resize((SIZE,SIZE))

        # We save the picture
        dir_save = my_dir_crop_resize + "/" + file_on_df # We go in the folder of the individual ind
        img_pixel.save(dir_save)


"""
Title: split_train_test_val

Description: split and copy the pictures in train / validation / test dataset for deep learning

Usage: split_train_test_val(my_dir_crop_resize,my_dir_split)

Argument: 
-	my_dir_crop_resize : directory where all the pictures are
-   my_dir_split : directory where all the pictures will be splitted

Details: Split and copy the pictures in train / validation / test dataset for deep learning

Values: /
"""


def split_train_test_val(my_dir_crop_resize,my_dir_split):
    
    
    # ======= I/ DATAFRAME INDIVIDUAL / PATH =============================
    

    # Crop + resize pictures : folder with all the pictures of the population cropped
    print("The picture directory is ",my_dir_crop_resize)

    # Number of individuals we have :
    nb_individual = len(os.listdir(my_dir_crop_resize))
    print("There are ",nb_individual," individuals\n")
    
    dataset_ready = []

    # We iterate over the folders (individuals)
    for indiv in os.listdir(my_dir_crop_resize):
      for picturename in os.listdir(os.path.join(my_dir_crop_resize, indiv)):
        dataset_ready.append({'individual':indiv, 'path': os.path.join(my_dir_crop_resize, indiv, picturename)})

    # We prepare a dataframe and preview it
    dataset_ready = pd.DataFrame(dataset_ready)
    print(dataset_ready.head(10))

    # https://stackoverflow.com/questions/50748511/pandas-dataframe-count-occurrence-of-list-element-in-dataframe-rows  
    # Let's analyse that dataset
    print("\nThe shape of the dataframe is ",dataset_ready.shape[0]," lines and ",dataset_ready.shape[1], "columns")
    print("The minimal picture number by class is : ",np.min(dataset_ready["individual"].value_counts())) # Minimal number of picture for the classes
    print(dataset_ready.individual.value_counts()) # Show number of picture by individual
    
    
    
    # ======= II/ SPLIT IN TRAIN AND TEST AND VALIDATION =========================

    # Dataset split in train / test
    # - test_size = % du dataset test
    # - stratify = If not None, data is split in a stratified fashion, using this as the class labels.
    print("df_train, df_test")
    df_train, df_test = train_test_split(dataset_ready, test_size=0.3, stratify=dataset_ready["individual"], shuffle=True)
    
    print("\ndf_test before separation in test / val:")
    df_test = df_test.reset_index(drop=True)
    print(df_test.head(10))
    
    print("df_train, df_val")
    # We cannot use stratify for the validation dataset, because too small
    df_train, df_val = train_test_split(df_train, test_size=0.1, shuffle=True)
    
    # And we reset the index
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    
    # ======= III/ MOVE TESTING INDIVIDUALS WITH 1 PICTURES ===================

    # If we have individuals in the test dataset with only 1 picture, we put it in the train dataset
    # As we have coded, if we have only 1 picture for an individual, we can't check if the closest picture is the same individual
    # We need at least 2 pictures by individual in the test dataset
    
    # Find individuals with 1 pictures in the test dataset
    ind_test_1pic = list(df_test["individual"].value_counts()[df_test["individual"].value_counts() == 1].index)
    print("Here the individuals in the test dataset with 1 picture")
    print(ind_test_1pic)

    # Then the index : we obtain a vector of all the lines to move
    idx = []
    for i in range(0,len(df_test)):
        if df_test.individual.iloc[i] in ind_test_1pic:
            idx = idx + [i]
    print("The final index is ")
    print(idx)

    # We add these lines to the train dataframe
    df_train = pd.concat([df_train,df_test.iloc[idx]])
    # And we reset the index
    df_train = df_train.reset_index(drop=True)
    print(df_train.head())

    # Then we delete these lines in the test dataframe
    df_test = df_test.drop(idx)
    # And we reset the index
    df_test = df_test.reset_index(drop=True)
    print(df_test.head())
    
    
    # ======= IV/ WE PREPARE THE DATASETS =====================================
    # -- 1) Train dataset ------------------------------------------------------

    # Let's analyse that train dataset
    print("\nThe shape of the train dataframe is ",df_train.shape[0]," lines and ",df_train.shape[1], "columns") 
    print("The minimal picture number by class is : ",np.min(df_train["individual"].value_counts())) # Minimal number of picture for the classes

    # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python (get unique value)
    ind_train_unique=list(df_train["individual"])
    ind_train_unique=set(ind_train_unique)
    ind_train_unique=list(ind_train_unique)
    print("Here the individuals of the train dataset : ",ind_train_unique)

    print("")
    print(df_train.individual.value_counts()) # Show number of picture by individual
    
    
    # -- 2) Test dataset ------------------------------------------------------
    
    # Let's analyse that test dataset
    print("The shape of the test dataframe is ",df_test.shape[0]," lines and ",df_test.shape[1], "columns")  
    print("The minimal picture number by class is : ",np.min(df_test["individual"].value_counts())) # Minimal number of picture for the classes

    # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python (get unique value)
    ind_test_unique=list(df_test["individual"])
    ind_test_unique=set(ind_test_unique)
    ind_test_unique=list(ind_test_unique)
    print("Here the individuals of the test dataset : ",ind_test_unique)

    print("")
    print(df_test.individual.value_counts()) # Show number of picture by individual
    
    
    # -- 3) Validation dataset ------------------------------------------------------
    
    # Let's analyse that test dataset
    print("The shape of the test dataframe is ",df_val.shape[0]," lines and ",df_val.shape[1], "columns")  
    print("The minimal picture number by class is : ",np.min(df_val["individual"].value_counts())) # Minimal number of picture for the classes

    # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python (get unique value)
    ind_val_unique=list(df_val["individual"])
    ind_val_unique=set(ind_val_unique)
    ind_val_unique=list(ind_val_unique)
    print("Here the individuals of the test dataset : ",ind_val_unique)

    print("")
    print(df_val.individual.value_counts()) # Show number of picture by individual
    
    
    # ======= V/ TRANSFER IN THE TRAIN OR TEST OR VALIDATION FOLDER ==============
    # -- 1) Create the folders ------------------------------------------------
    
    
    # We create the main folder
    
    os.makedirs(my_dir_split)
    print("The path with all the individuals splitted is ",my_dir_split)
    
    # Let's create the train / test / validation folders thanks to the dataframe
    
    dir_train = os.path.join(my_dir_split, "train")
    os.makedirs(dir_train)

    dir_test = os.path.join(my_dir_split, "test")
    os.makedirs(dir_test)
    
    dir_val = os.path.join(my_dir_split, "validation")
    os.makedirs(dir_val)
    
    
    # -- 2) Copy all the labels (so sub-folders)-------------------------------

    print("Here the individuals of the train dataset : ",ind_train_unique)
    for i in range(len(ind_train_unique)):    
        newpath_folder_train = os.path.join(dir_train,ind_train_unique[i])
        os.makedirs(newpath_folder_train)

    print("Here the individuals of the test dataset : ",ind_test_unique)
    for i in range(len(ind_test_unique)):
        newpath_folder_test = os.path.join(dir_test,ind_test_unique[i])
        os.makedirs(newpath_folder_test)
        
    print("Here the individuals of the validation dataset : ",ind_val_unique)
    for i in range(len(ind_val_unique)):
        newpath_folder_val = os.path.join(dir_val,ind_val_unique[i])
        os.makedirs(newpath_folder_val)
        
    
    
    # -- 3) Copy all the pictures ---------------------------------------------
    
    print("we copy the train dataset")
    for i in range(len(df_train)):

        # The path of the source of the picture :
        source = df_train.path[i] # For example 'D:/deep-learning_re-id_gimenez\dataset_test2_crop_resize\Kely\kely3.jpg'
        
        # The name of the picture :
        pic_name = os.path.basename(df_train.path[i]) # For example 'kely3.jpg'

        # The path of the destination of the picture :
        # dir_train is for example D:/deep-learning_re-id_gimenez\dataset_ready/test'
        destination = os.path.join(dir_train, df_train.individual[i],pic_name) # For example 'D:/deep-learning_re-id_gimenez\dataset_ready/test\Kely\kely3.jpg'

        # Then we copy
        shutil.copyfile(source,destination)

    print("we copy the test dataset")
    for i in range(len(df_test)):

        # The path of the source of the picture :
        source = df_test.path[i] 

        # The name of the picture :
        pic_name = os.path.basename(df_test.path[i]) 

        # The path of the destination of the picture :
        destination = os.path.join(dir_test, df_test.individual[i],pic_name) 

        # Then we copy
        shutil.copyfile(source,destination)

    print("we copy the validation dataset")
    for i in range(len(df_val)):

        # The path of the source of the picture :
        source = df_val.path[i] 

        # The name of the picture :
        pic_name = os.path.basename(df_val.path[i])

        # The path of the destination of the picture :
        destination = os.path.join(dir_val, df_val.individual[i],pic_name) 

        # Then we copy
        shutil.copyfile(source,destination)





"""
Title: augmentation

Description: augment the train dataset

Usage: augmentation(nb_pic_aug, my_dir_augmentation, my_dir_split)

Argument: 
-	nb_pic_aug : number of pictures we want in total for each individuam
-   my_dir_augmentation : directory of the folder augmented
-   my_dir_split : directory of the splitted pictures

Details: augment the train dataset by data augmentation

Values: /
"""




def augmentation(nb_pic_aug, my_dir_augmentation, my_dir_split):

    # ======= I/ CREATING AND COPYING THE FOLDERS / IMAGES ====================
    
    # We create the mother folder with all the augmentation pictures
    os.makedirs(my_dir_augmentation)
    
    # We write the folder with the train dataset
    my_dir_split_train = os.path.join(my_dir_split, "train")

    # We need first to copy all the folder names with the labels of the individuals to the new augmentation folder
    # We take the list of the folders names = list of the individual names
    labels_ind = os.listdir(my_dir_split_train)
    print("Here the different labels ", labels_ind, " \n")

    # We store the picture path
    destination_all = []

    # For each label
    for ind in range(len(labels_ind)):

        destination_ind = []

        # We copy the folder
        newpath = my_dir_augmentation + "/" + labels_ind[ind]
        os.makedirs(newpath)
        print("The new folder is ", newpath)

        # We copy all the pictures of the individual ind
        images_ind = os.listdir(os.path.join(
            my_dir_split_train, labels_ind[ind]))
        print(images_ind)
        for pic in range(len(images_ind)):

            # The path of the source of the picture :
            source = os.path.join(my_dir_split_train,
                                  labels_ind[ind], images_ind[pic])
            print(source)

            # The path of the destination of the picture :
            # For example 'D:/deep-learning_re-id_gimenez\dataset_ready/test\Kely\kely3.jpg'
            destination = os.path.join(
                my_dir_augmentation, labels_ind[ind], images_ind[pic])
            destination_ind = destination_ind + [destination]
            print(destination)

            # Then we copy
            shutil.copyfile(source, destination)

        # We have [[picture1_individual1, picture2_individual1, ...],...,[picture1_individual99,picture2_individual99,...]]
        # Separating the path between individuals will be useful for the data augmentation to have the same amount of picture for each augmented folders
        destination_all.append(destination_ind)

    print(destination_all)



    # ======= II/ AUGMENT THE PICTURES =========================================
    
    # We augment the image with different methods, keeping the augmentation realistic, using a sequence
    
    # sometimes = applies the given augmenter in 50% of all cases
    # https://gitee.com/alavaien/imgaug
    def sometimes(aug): return iaa.Sometimes(0.5, aug)
    
    seq = iaa.Sequential([

        # ROTATION : from -45° to 45°
        # https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html
        iaa.Rotate((-45, 45)),

        # BLUR : scale between 2 and 4
        # k = Kernel size to use (If a tuple of two int (a, b), then the kernel size will be sampled from the interval [a..b])
        # https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html
        sometimes(iaa.AverageBlur(k=(2, 4))),
        
        # GRAYSCALE
        # Augmenter to convert images to their grayscale versions
        # Change images to grayscale and overlay them with the original image by varying strengths, effectively removing 0 (alpha=0) to 100% (alpha=1) of the color
        # https://imgaug.readthedocs.io/en/latest/source/overview/color.html#grayscale
        
        sometimes(iaa.Grayscale(alpha=1.0)),

        # BRIGHTNESS
        # Add -50 to 50 to the brightness-related channels of each image
        sometimes(iaa.WithBrightnessChannels(iaa.Add((-50, 50))))
        
    ], random_order=True)  # apply augmenters in random order

    # We then augment all the pictures : (x nb_augment_by_pic)

    # We want to augment the pictures for the individuals BUT to have the same amount of pictures for each individual at the end
    # It is better for the deep learning classification procedure

    print("We want", nb_pic_aug, "by individual after the augmentation")

    # FOR EACH INDIVIDUAL
    for ind in range(len(destination_all)):

        # nb_pic_aug : total number of picture we want for each individual by augmentation
        # nb_pic_aug = total_pic_ind + (total_pic_ind - 1) * nb_pic_min + nb_pic_extra
        # the number of picture for this individual
        total_pic_ind = len(destination_all[ind])
        # Number of minimal augmentation by picture = the result of the euclidian division. We put - 1 because we need to count the original pictures
        nb_pic_min = nb_pic_aug//total_pic_ind - 1
        nb_pic_extra = nb_pic_aug % total_pic_ind  # the remeaning of the division

        # Example :
        # We want 50 augmented pictures by individual. We have initially 19 individuals.
        # 50 // 19 - 1 = 1
        # 50 % 19 = 12
        # nb_aug_by_pic = [2,2,2,2,2,2,2,...,1,1,1,1,1,1,1]
        # We have 12 x 2 pictures augmented, and 7 x 1 pictures augmented = 31 pictures augmented
        # And we have the 19 original pictures, so 19 + 31 = 50, everything is good
        nb_aug_by_pic = [(nb_pic_min+1) for i in range(nb_pic_extra)] + \
            [nb_pic_min for i in range(total_pic_ind-nb_pic_extra)]
        print("Here the augmentation for each picture of the individual ",
              ind, " : ", nb_aug_by_pic)

        # FOR EACH PICTURE OF THIS INDIVIDUAL :
        for pic in range(len(destination_all[ind])):

            # We first read the image
            path_load = str(destination_all[ind][pic])
            print("ind is ", ind)
            print("pic is ", pic)
            print("path image i is : ", path_load)
            img = imageio.imread(path_load)

            # The number of augmentation for this picture
            nb_aug_pic = nb_aug_by_pic[pic]
            print("There is ", nb_aug_pic, " augmentation for this picture")

            # FOR EACH AUGMENTATION OF THIS PICTURE :
            for nb in range(nb_aug_pic):

                print("nb is ", nb)

                # Then augment it
                img_aug = seq(image=img)

                # Then we save the image
                individual = os.path.basename(
                    path_load)  # Give the individual name
                print("individual is ")
                print(individual)
                print("individual[-4:]")
                print(individual[-4:])
                
                # We have some problems with the picture format (jpg or jpeg)
                # if individual[-4:] == ".jpg" or individual[-4:] == ".JPG" or individual[-4:] == ".JPG":
                #     path_save = path_load[:(len(path_load)-4)] + \
                #         "_aug" + str(nb) + individual[-4:]
                #     print("path save is ", path_save)
                if individual[-4:] == "jpeg":
                    path_save = path_load[:(len(path_load)-4)] + \
                        "_aug" + str(nb) + individual[-5:]
                    print("path save is ", path_save)
                else:
                    print("not jpeg")
                    path_save = path_load[:(len(path_load)-4)] + \
                        "_aug" + str(nb) + individual[-4:]
                    print("path save is ", path_save)
                    
                plt.imsave(path_save, img_aug)





"""
Title: resize_all_pic_all_files

Description: Copy and resize a whole folder of splitted folders of individuals with pictures

Usage: resize_all_pic_all_files (my_dir_pre_processing,new_size)

Argument: 
-	my_dir_pre_processing : directory of all the folders / files to copy and resize
-   new_size : the new size we want

Details:  Copy and resize a whole folder of splitted folders of individuals with pictures

Values: /
"""




def resize_all_pic_all_files (my_dir_pre_processing,new_size):
    

    # ======= I/ CREATING AND COPYING THE FOLDERS =============================
    
    
    # We create the folder with all the crop resize pictures
    my_dir_new = os.path.join(my_dir_pre_processing,"dataset_split_"+str(new_size))
    os.makedirs(my_dir_new)
    
    my_dir_old = os.path.join(my_dir_pre_processing,"dataset_split_300")
    
    all_files = os.listdir(my_dir_old)
    print("Here all the files to resize : ",all_files)
    
    for my_file in all_files :
        
        print("my_file is ",my_file)
    
        # We need first to copy all the folder names with the labels of the individuals to the new crop folder
        labels_ind = os.listdir(os.path.join(my_dir_old,my_file)) # We take the list of the folders names = list of the individual names
        print("Here the different labels ",labels_ind)
        for i in range(len(labels_ind)):
            newpath = os.path.join(my_dir_new,my_file,labels_ind[i])
            os.makedirs(newpath)
            
            
        # ======= II/ RESIZE THE PICTURES =====================================
        
        # https://www.askpython.com/python/examples/crop-an-image-in-python (crop an image)  
        # https://stackoverflow.com/questions/6444548/how-do-i-get-the-picture-size-with-pil (get picture size)  
        
        # For each individual :
        for ind in labels_ind:
            
            print("ind is ",ind)
            
            list_pictures = os.listdir(os.path.join(my_dir_old,my_file,ind))
            print("list pictures is")
            print(list_pictures)
            
            for pic in list_pictures:
                
                dir_load = os.path.join(my_dir_old,my_file,ind,pic)
                
                img = Image.open(dir_load) 
                img = img.convert('RGB') # necessary if we have transparent pictures (https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg)
                
                # We resize the picture
                img = img.resize((new_size,new_size))
        
                # We save the picture
                dir_save = os.path.join(my_dir_new,my_file,ind,pic) # We go in the folder of the individual ind
                img.save(dir_save)




# =============================================================================
# ===  MAIN FUNCTION ==========================================================
# =============================================================================


def preprocessing(nb_pic_aug,my_dir_augmentation,my_dir_pop,my_megadetect_path,my_dir_main,my_dir_bbox_pb,treshold_bbox,my_dir_crop_resize, my_folder_raw_animal,my_dir_split):
    
    
    # ======= I/ DELETE ANUMAL FOLDERS WITH 1 PICTURES ========================

    delete_pic_1_folder(my_dir_pop)
    
    # ======= II/ OBTAIN A .JSON FILE WITH MEGADETECTOR =======================
    
    """
    Here the explaination to obtain a .json file with the Megadetector :
      -> https://github.com/microsoft/CameraTraps/blob/main/megadetector.md
    The goal of Megadetector in our case is to have the coordinates of the bbox around the animal
    """

    # ======= III/ FROM .JSON FILE TO A DATAFRAME ===============================
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True # For truncated pictures (https://stackoverflow.com/questions/60584155/oserror-image-file-is-truncated)
    
    df_json_final = from_json_to_df(my_megadetect_path,my_dir_main,my_dir_bbox_pb,treshold_bbox)

    ind_step1_unique2=list(df_json_final["file"])
    ind_step1_unique2=set(ind_step1_unique2)
    ind_step1_unique2=list(ind_step1_unique2)
    print(ind_step1_unique2)

    print(len(ind_step1_unique2))
    
    
    # ======= IV/ CROP PICTURES AND RESIZE THEM THANKS TO THE DATAFRAME ============

    crop_resize(my_dir_crop_resize, my_dir_main, my_folder_raw_animal,df_json_final)

    
    # ======= V/ SPLIT TRAIN TEST VALIDATION DATASETS ==========================

    split_train_test_val(my_dir_crop_resize, my_dir_split)

    # ======= VI/ DATA AUGMENTATION ============================================

    augmentation(nb_pic_aug, my_dir_augmentation, my_dir_split)


# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

# SIZE : size of the pictures
SIZE = 300 

# MEGADETECTOR_PATH : path of the megadetector file
MEGADETECTOR_PATH = "1_Pre-processing/megadetector/output_3ind.json"

# NB_PIC_AUGMENT : the number of picture you want for each individual. Necessary to choose it using the first step.
NB_PIC_AUGMENT = 200


# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# MAIN DIRECTORY : Where we have everything except the code
dir_main = 'D:/deep-learning_re-id_gimenez'
print("The main directory is ",dir_main,"\n")

# FOLDER RAW : Folder with all the raw pictures of the population
folder_raw_animal = os.path.join("0_dataset_Marie","0_dataset_Marie_3ind")
dir_pop = os.path.join(dir_main, folder_raw_animal)
print("The directory with all the individuals is ",dir_pop,"\n")

# FOLDER BBOX PROBLEMS : Folder with all the bbox problems
folder_bbox_pb = 'dataset_bbox_pb'
dir_bbox_pb = os.path.join(dir_main,"1_Pre-processing", folder_bbox_pb)
print("The directory with all the bbox problems is ",dir_bbox_pb,"\n")

# FOLDER READY : Folder with all the pictures of the cropped and resized population
folder_ready_animal = 'dataset_ready'
dir_pop_crop_resize = os.path.join(dir_main,"1_Pre-processing", folder_ready_animal)
print("The directory of the crop resized pictures is ",dir_pop_crop_resize,"\n")

# FOLDER SPLIT : Folder with split images which will be used for the deep-learning
folder_animal_split = 'dataset_split'
dir_split = os.path.join(dir_main, "1_Pre-processing", folder_animal_split)
print("The directory of the splitted pictures is ",dir_split,"\n")

# FOLDER DATA AUGMENTATION : Data augmentation of the pictures :
folder_augmentation = "train_augmentation"
dir_augmentation = os.path.join(dir_split, folder_augmentation)
print("The directory of the data augmentation pictures is ", dir_augmentation, "\n")


# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

# ======= I/ PREPARE PICTURES FOR 1 MODEL =====================================

preprocessing(nb_pic_aug = NB_PIC_AUGMENT,my_dir_augmentation = dir_augmentation,my_dir_pop = dir_pop,my_megadetect_path = MEGADETECTOR_PATH, my_dir_main = dir_main, my_dir_bbox_pb = dir_bbox_pb, treshold_bbox = 0.23, my_dir_crop_resize = dir_pop_crop_resize, my_folder_raw_animal = folder_raw_animal,my_dir_split = dir_split)


# ======= II/ PREPARE OTHER DATASETS WITH DIFFERENT SIZES FOR OTHER MODELS ========

# Then we copy theses files in different sizes to test different models
# All the size to test : 224, 240, 260, 300, 380, 456, 528, 600 (EfficientNetB0 to B7)
ALL_SIZE = [224, 260] # We don't write 300 because it's the original size chosen
for my_size in ALL_SIZE:
    resize_all_pic_all_files("D:/deep-learning_re-id_gimenez/1_Pre-processing",my_size)




