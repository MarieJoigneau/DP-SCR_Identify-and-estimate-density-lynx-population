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

import os
from PIL import Image
from PIL import ImageFile, Image


# =============================================================================
# =============================================================================
# ======================== FUNCTIONS ==========================================
# =============================================================================
# =============================================================================



"""
Title: resize_all_pic_all_files

Description: Copy and resize a whole folder of splitted folders of individuals with pictures

Usage: resize_all_pic_all_files (my_dir_pic_OFB,new_size)

Argument: 
-	my_dir_pic_OFB : directory of all the folders / files to copy and resize
-   my_new_size : the new size we want

Details:  Copy and resize a whole folder of splitted folders of individuals with pictures

Values: /
"""


def resize_all_pic_all_files (my_dir_pic_OFB,my_new_size):
    

    # ======= I/ CREATING AND COPYING THE FOLDERS =============================
    
    
    # We create the folder with all the crop resize pictures
    my_dir_new = os.path.join(my_dir_pic_OFB,"dataset_ready")
    os.makedirs(my_dir_new)
    
    # The directory of the raw dataset
    my_dir_old = os.path.join(my_dir_pic_OFB,"dataset_raw")
    
    
    # We need first to copy all the folder names with the labels of the individuals to the new crop folder
    labels_ind = os.listdir(my_dir_old) # We take the list of the folders names = list of the individual names
    print("Here the different labels ",labels_ind)
    for i in range(len(labels_ind)):
        newpath = os.path.join(my_dir_new,labels_ind[i])
        os.makedirs(newpath)
        
        
    # ======= II/ RESIZE THE PICTURES =====================================
    
    # https://www.askpython.com/python/examples/crop-an-image-in-python (crop an image)  
    # https://stackoverflow.com/questions/6444548/how-do-i-get-the-picture-size-with-pil (get picture size)  
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True # For truncated pictures (https://stackoverflow.com/questions/60584155/oserror-image-file-is-truncated)
    
    # For each individual :
    for ind in labels_ind:
        
        print("ind is ",ind)
        
        list_pictures = os.listdir(os.path.join(my_dir_old,ind))
        print("list pictures is")
        print(list_pictures)
        
        for pic in list_pictures:
            
            dir_load = os.path.join(my_dir_old,ind,pic)
            
            img = Image.open(dir_load) 
            img = img.convert('RGB') # necessary if we have transparent pictures (https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg)
            
            # We resize the picture
            img = img.resize((my_new_size,my_new_size))
    
            # We save the picture
            dir_save = os.path.join(my_dir_new,ind,pic) # We go in the folder of the individual ind
            img.save(dir_save)



# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

# SIZE : size of the pictures
SIZE = int(input("Write here the size you want / Ecrivez ici la taille que vous desirez ")) # 260

# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================


dir_pic_OFB = str(input("Write here the directory of the OFB folder (end by '/OFB lynx' or '/OFB jaguar') / Ecrivez ici le chemin d’acces du dossier OFB (fini par '/OFB lynx' ou '/OFB jaguar') ")) # D:/OneDrive/Stage M2 - Montpellier Gimenez/OLIVIER/A envoyer OFB lynx/Deep learning lynx - prediction/OFB lynx


# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

resize_all_pic_all_files (my_dir_pic_OFB = dir_pic_OFB, my_new_size = SIZE)








