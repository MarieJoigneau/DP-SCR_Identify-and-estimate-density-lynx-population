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
Title: from_json_to_df

Description: takes the output of Megadetector which focus the animal in the picture
and converts it into a dataframe

Usage: from_json_to_df(my_megadetect_path, my_dir_main, treshold_bbox)

Argument: 
-	my_megadetect_path : directory where the Megadetector file is (JSON file)
-   my_dir_main : main directory of the pictures
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


def from_json_to_df (my_megadetect_path, my_dir_main, treshold_bbox):
    
    
    
    # ======= I/ OBTAINING A FIRST DATAFRAME ==================================
    # -- 1) A dataframe with 1 line -------------------------------------------
    print("======= I/ OBTAINING A FIRST DATAFRAME ==================================")
    
    
    # https://www.geeksforgeeks.org/read-json-file-using-python/ 

    # We open the JSON file
    f = open(my_megadetect_path)

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
    
    auto_or_manual = input("do you want to filter automatically? (yes/no) / voulez-vous filtrer automatiquement? (yes/no) ")
    
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
        my_dir_bbox_pb = os.path.join(my_dir_main,"bbox_pb")
        os.makedirs(my_dir_bbox_pb)
    
        path_bbox_problem = []
    
        # Then we do bbox crop with it. We will crop all the pictures :
        for i in range(0,df_step2.shape[0]):
    
            # We find the picture :
            file_on_df = df_step2.file[i]
            print(file_on_df)
            dir_ind = os.path.join(my_dir_main,"dataset_raw",file_on_df) # We go in the folder of the individual ind
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
        following = input("Note the pictures you want to keep, and when you are ready, enter a letter / Notez les images à garder et écrivez une lettre quand vous êtes prêts")
        
        
        
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

Usage: crop_resize (my_dir_crop_resize,my_dir_main,my_df_json_final)

Argument: 
-	my_dir_crop_resize : directory where all the pictures are
-   my_dir_main : main directory
-   my_df_json_final : dataframe with coordinates of bounding boxes of the pictures

Details: 
    
Takes the output of Megadetector which focus the animal in the picture

Values: /
"""



def crop_resize (my_dir_main,my_df_json_final):
    
    my_dir_crop_resize = os.path.join(my_dir_main,"dataset_ready")
    
    # ======= I/ CREATING AND COPYING THE FOLDERS =============================
    
    # We create the folder with all the crop resize pictures
    os.makedirs(my_dir_crop_resize)
    
    # We need first to copy all the folder names with the labels of the individuals to the new crop folder
    labels_ind = os.listdir(os.path.join(my_dir_main,"dataset_raw")) # We take the list of the folders names = list of the individual names
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
        dir_ind = os.path.join(my_dir_main, "dataset_raw",file_on_df) # We go in the folder of the individual ind
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



def preprocessing_crop_resize(my_megadetect_path,my_dir_main,treshold_bbox):
    
   
    # ======= I/ OBTAIN A .JSON FILE WITH MEGADETECTOR =======================
    
    """
    Here the explaination to obtain a .json file with the Megadetector :
      -> https://github.com/microsoft/CameraTraps/blob/main/megadetector.md
    The goal of Megadetector in our case is to have the coordinates of the bbox around the animal
    """

    # ======= III/ FROM .JSON FILE TO A DATAFRAME ===============================
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True # For truncated pictures (https://stackoverflow.com/questions/60584155/oserror-image-file-is-truncated)
    
    df_json_final = from_json_to_df(my_megadetect_path = MEGADETECTOR_PATH, my_dir_main = dir_pic_OFB,treshold_bbox = 0.23)

    ind_step1_unique2=list(df_json_final["file"])
    ind_step1_unique2=set(ind_step1_unique2)
    ind_step1_unique2=list(ind_step1_unique2)
    print(ind_step1_unique2)

    print(len(ind_step1_unique2))
    
    
    # ======= IV/ CROP PICTURES AND RESIZE THEM THANKS TO THE DATAFRAME ============

    crop_resize(my_dir_main,df_json_final)


# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

# SIZE : size of the pictures
SIZE = int(input("Write here the size you want / Ecrivez ici la taille que vous desirez")) # 260


# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# "D:/OneDrive/Stage M2 - Montpellier Gimenez/OLIVIER/A envoyer OFB lynx/Deep learning lynx - prediction/OFB lynx"
# D:/OneDrive/Stage M2 - Montpellier Gimenez/OLIVIER/A envoyer OFB jaguar/Deep learning jaguar - prediction/OFB jaguar
dir_pic_OFB = str(input("Write here the directory of the OFB folder (end by '/OFB lynx' or ‘OFB jaguar’)  / Ecrivez ici le chemin d’acces du dossier OFB (fini par '/OFB lynx' ou '/OFB jaguar') ")) 


# MEGADETECTOR_PATH : path of the megadetector file
MEGADETECTOR_PATH = os.path.join(dir_pic_OFB,"megadetector_results.json")

# Megadetector detect several bounding boxes with different confidence. 
# You can erase some boxes depending of their confidence with the treshold. 
# If confidence > treshold, you will have the boxe in the selection for you to choose
MY_TRESH_BBOX = float(input("Which treshold you want for the bounding boxes? 0.26 recommanded / Quel seuil voulez vous pour les boîtes qui cadrent ? 0.26 recommandé"))

# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

preprocessing_crop_resize(my_megadetect_path = MEGADETECTOR_PATH, my_dir_main = dir_pic_OFB,treshold_bbox = MY_TRESH_BBOX)









