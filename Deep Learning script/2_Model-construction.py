# -*- coding: utf-8 -*-
"""
PROJET DEEP LEARNING LYNX - Internship M2 2023

Author : Marie Joigneau
Supervisor : Olivier Gimenez

Problematic : Re-identify individual and estimate population density from eurasian lynx population

Before : 
- We have prepared the pictures for the model

This part : Here we will construct different models depending of different parameters,
to be tested after to choose finally the best model
"""

# =============================================================================
# =============================================================================
# ======================== PACKAGES ===========================================
# =============================================================================
# =============================================================================

import pandas as pd
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import keract
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf
from kerasgen.balanced_image_dataset import balanced_image_dataset_from_directory


# =============================================================================
# =============================================================================
# ======================== FUNCTIONS ==========================================
# =============================================================================
# =============================================================================

"""
Title: lrfn

Description: define the Learning Rate schedule as a callback
(Callback = permit to save the parameters)
(LR : https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html)

Usage: lrfn(epoch)

Argument: 
-	epoch (number of epoch)

Details: 
Define the LR schedule as a callback
Set values for starting, maximum and minimum learning rate possible
Set number of epochs to rampup and sustain learning rates along with learning rate decay factor

Values: 
-   start_lr : Set values for starting learning rate possible
-   min_lr : Set values for minimum learning rate possible
-   min_lr : Set values for maximum learning rate possible
-   rampup_epochs : Set number of epochs to ramp up learning rates
-   sustain_epochs : Set number of epochs to sustain learning rates
-   exp_decay : learning rate decay factor
(https://freecontent.manning.com/fine-tuning-a-pre-trained-resnet-50/)
"""

def lrfn(epoch):
    
    # Define the LR schedule constants = define configuration parameters :
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.00005
    rampup_epochs = 5
    sustain_epochs = 0
    exp_decay = .8
    
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch -
                                                 rampup_epochs-sustain_epochs) + min_lr
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)




"""
Title: construct_model

Description: construct a deep learning model

Usage: construct_model(my_premodel, my_embedding_size, my_triplet_choice, my_epoch_number, my_train_ds, my_val_ds)

Argument: 
-	my_premodel (the model transfered by transfer learning to be the base)
-   my_embedding_size (size of the vector in output)
-	my_triplet_choice (metric of the triplet loss function)
-   my_epoch_number (number of epoch we want)
-   my_train_ds (dataset to train the model)
-   my_val_ds (dataset to validate the model)

Details: 
Construct a deep learning model depending of several layers

Values: 
-   my_model : the deep learning model (keras)
-   my_history : recording of the steps to create the model
"""


def construct_model(my_premodel, my_embedding_size, my_triplet_choice, my_epoch_number, my_train_ds, my_val_ds,my_first_dense_size):
    
    # ======= I/ PRE-MODEL SETTING ============================================
    
    # We want the pre-model to be also trainable when we train it on each epoch to minimize the loss function
    my_premodel.trainable = True


    # ======= II/ CONSTRUCTION OF THE FINAL LAYERS ============================
    # -- 1) First half of the layers ------------------------------------------

    # Create the placeholder for the input image and run it through our pre-trained VGG16 model.
    # Note that we will be fine-tuning this feature extractor but it will still run in *inference model* making sure the
    # BatchNorm layers are not training. 

    # Placeholder : stands for the class-specific threshold between known and unknown for open-set recognition
    # (Zhou et al. - 2021 - Learning Placeholders for Open-Set Recognition.pdf or https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Learning_Placeholders_for_Open-Set_Recognition_CVPR_2021_paper.pdf)

    # BatchNorm layers : BatchNormalization layer
    # - Layer that normalizes its inputs.
    # - Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
    # - Works differently during training and during inference
    # (https://keras.io/api/layers/normalization_layers/batch_normalization/)

    # INPUT LAYER:
    # shape = [(None, 224, 224, 3)] = (pixel,pixel,RGB) = (224,224,3)
    # parameters = 0
    inputs = keras.layers.Input(shape=my_premodel.input_shape[1:])

    # FEATURES :
    # vgg16 (Functional) = we take features of the pre-trained model (transfert learning)
    # shape = (None, 7, 7, 512)
    # parameters = 14714688
    features = my_premodel(inputs, training=False)

    
    # -- 2) Upper half of the layers ------------------------------------------
    
    # Create the upper half of our model where we pool out the extracted features, pass it through a mini fully-connected
    # network. We also force the embeddings to remain on a unit hypersphere space. (Mathias)

    # LAYER 1 : Global Average Pooling (2D)
    # - shape = (None, 512)
    # - parameters = 0
    # => Global Average Pooling is a pooling operation designed to replace fully connected layers in classical CNNs
    # (https://paperswithcode.com/method/global-average-pooling)
    # => Example with explaination (https://www.youtube.com/watch?v=gNRVTCf6lvY)
    # It converts all the n features maps (dimension x*y) to a flatten layer (dimension n*1)
    x = keras.layers.GlobalAveragePooling2D()(features)

    # LAYER 2 : Dense
    # - shape = (None, 2048)
    # - parameters = 1050624
    # => relu : rectified linear units (ReLUs) = if number positive, keep it. Otherwise you get a zero
    # Dense layer : reduce the dimension of the input layer (n neurons) to the output layer (n-k neurons)
    x = keras.layers.Dense(my_first_dense_size, activation='relu')(x)  # 2048 neurons, activation function = relu

    # LAYER ? : Dropout
    #x = keras.layers.Dropout(dropout)(x)

    # LAYER 3 = Dense
    # - shape = (None, 128)
    # - parameters = 262272
    # => Embedding : we translate high-dimensional vectors into a relatively low-dimensional space
    # (https://towardsdatascience.com/what-is-embedding-and-what-can-you-do-with-it-61ba7c05efd8)
    x = keras.layers.Dense(my_embedding_size)(x)

    # OUTPUT LAYER = lambda
    # We normalize = checked on github, even cosine need l2_normalization (Mathias)
    outputs = keras.layers.Lambda(lambda a: tf.math.l2_normalize(a, axis=1))(x)

    
    # -- 3) And so the model --------------------------------------------------

    # We create the final model
    my_model = keras.models.Model(inputs, outputs)
    my_model.summary()
    

    # ======= III/ MODEL COMPILATION ==========================================

    # Adam optimizer :
    # - Adaptive Moment Estimation
    # - Makes use of an exponentially decaying average of past gradients
    # - Employs an exponentially decaying average of past squared gradients in order to provide an adaptive learning rate.
    # - Currently the Adam optimizer is the preferred optimizer for use with deep learning models.
    # (https://machinelearningjourney.com/index.php/2021/01/09/adam-optimizer/)

    # Compile the model with TripletLoss and Adam optimizer
    my_model.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tfa.losses.TripletSemiHardLoss(distance_metric=my_triplet_choice))


    # ======= IV/ CALLBACK ====================================================
    
    # Define the learning rate
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: lrfn(epoch), verbose=True)

    # Define a `EarlyStopping` callback so that our model does overfit
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=2, mode='auto',
        restore_best_weights=True
    )

    # ======= V/ TRAINING PROGRESS ============================================

    # Train the model and visualize the training progress
    my_history = my_model.fit(my_train_ds,
                              validation_data=my_val_ds,
                              epochs=my_epoch_number,
                              callbacks=[lr_callback, es])

    return my_model, my_history



"""
Title: plot_training

Description: plot the training and validation loss along the epochs

Usage: plot_training(H, embedding_dim, my_dir_history_plot)

Argument: 
-	H : history of the deep learning model
-   embedding_dim : dimension of the embedding
-   my_dir_history_plot : directory where we want to save the graph

Details: 
Plot the training and validation loss along the epochs

Values: 
-   my_model : the deep learning model (keras)
-   my_history : recording of the steps to create the model
"""

def plot_training(H, embedding_dim, my_dir_history_plot):

    # We create the figure
    plt.plot(H.history['loss'], label='train')
    plt.plot(H.history['val_loss'], label='validation')
    plt.legend(loc="lower left")

    # Then we save the plot
    plt.savefig(os.path.join(my_dir_history_plot, "history.jpg"), dpi='figure')


# =============================================================================
# ===  MAIN FUNCTION ==========================================================
# =============================================================================


"""
Title: modelconstruction

Description: Construct different types of models and save them

Usage: modelconstruction(my_dir_augmentation, my_dir_split_list,my_size_list, my_num_classes_per_batch_list, my_num_images_per_class_list,my_embedding_size,my_triplet_choice_list,my_epoch_number,my_path_image_test,my_dir_history_plot,my_dir_model,my_dir_mapping,my_mapping_or_not,my_folder_augmentation,my_premodel_type_list)

Argument: 
-	my_dir_augmentation : directory of the pictures to augment
-   my_dir_split_list : list of all the directories of the splitted pictures to be used for the different models
-   my_size_list : list of all the sizes to test
-   my_num_classes_per_batch_list : list of the number of class by batch
-   my_num_images_per_class_list : list of the number of image by class by batch
-   my_embedding_size : the embedding size
-   my_triplet_choice_list : the list of the metrics of the triplet loss to test
-   my_epoch_number : number of epoch in the model
-   my_path_image_test : path of the image which will be the example for the mapping
-   my_dir_history_plot : directory where the history plot will be save
-   my_dir_model : directory where the model will be save
-   my_dir_mapping : directory where the mapping will be save
-   my_mapping_or_not : if the user wants to have a mapping or not of the model features
-   my_folder_augmentation : the folder where the augmentation pictures are
-   my_premodel_type_list : list of the premodel to test (for the transfer learning)

Details: 
Construct different types of deep learning model depending on :
-   the size of the pictures
-   the number of class by batch
-   the number of image by class by batch
-   the embedding size
-   the number of epoch
-   the metric of the triplet loss
-   if we do a mapping or not
-   the premodel chosen for the transfer learning

Values: /
"""

def modelconstruction(my_dir_augmentation, my_dir_split_list,my_size_list, my_num_classes_per_batch_list, my_num_images_per_class_list,my_embedding_size_list,my_triplet_choice_list,my_epoch_number,my_path_image_test,my_dir_history_plot,my_dir_model,my_dir_mapping,my_mapping_or_not,my_folder_augmentation,my_premodel_type_list,my_first_dense_size_list):
    
    # We initialize the directory of the splitted pictures
    nb_dir_split = 0
    
    print("The list of my different size to test is ",my_size_list)
    print("The list of my splitted pictures directories to test is ",my_dir_split_list)
    
    for my_size in my_size_list:
        
        # We take the folder with the pictures of the size chosen
        my_dir_split = my_dir_split_list[nb_dir_split]
        nb_dir_split = nb_dir_split + 1
        print("The directory of my splitted pictures is ",my_dir_split)
        print("The size of my pictures to test is ",my_size)
        
        
        # ======= I/ PREPARE THE TRAIN AND VALIDATION DATASETS ================
        # -- 1) The train / validation dataframe ------------------------------

        train_dataset = []
    
        # Iterate over the folders (individuals)
        for indiv in os.listdir(os.path.join(my_dir_split, my_folder_augmentation)):
            for picturename in os.listdir(os.path.join(my_dir_split, my_folder_augmentation, indiv)):
                train_dataset.append({'individual': indiv, 'path': os.path.join(
                    my_dir_split, my_folder_augmentation, indiv, picturename)})
    
        # Prepare a dataframe and preview it
        train_dataset = pd.DataFrame(train_dataset)
        train_dataset.head(10)
    
    
        # Let's analyse that train dataset
        print("The shape of the train dataset is ",
              train_dataset.shape[0], " lines and ", train_dataset.shape[1], "columns")
        print("The minimal picture number by class is : ", np.min(
            train_dataset["individual"].value_counts()))  # Minimal number of picture for the classes
    
        nb_individual = len(os.listdir(os.path.join(my_dir_split, my_folder_augmentation)))
        print("There are ", nb_individual, " individuals in the train dataset")
    
        ind_train_unique = list(train_dataset["individual"])
        ind_train_unique = set(ind_train_unique)
        ind_train_unique = list(ind_train_unique)
        print("Here the individuals of the train dataset : ", ind_train_unique)
    
        # Show number of picture by individual
        print(train_dataset.individual.value_counts())
    
    
        # -- 2) The train dataset train_ds ------------------------------------
    
        min_images_par_class = np.min(train_dataset["individual"].value_counts())
        print("In the training dataset, there are at least ", min_images_par_class, " images by class \n")
    
        # kerasgen package :
        #   - A Keras/Tensorflow compatible image data generator for creating balanced batches.
        #   - This datagenerator is compatible with TripletLoss as it guarantees the existence of postive pairs in every batch.
        # (https://github.com/ma7555/kerasgen)
        
        # We want to have different models depending of nb of class / nb of images per class
        for i in range(len(my_num_classes_per_batch_list)):
            
            my_num_classes_per_batch = my_num_classes_per_batch_list[i]
            my_num_images_per_class = my_num_images_per_class_list[i]
            print("There are ",my_num_classes_per_batch," classes by batch")
            print("There are ",my_num_images_per_class," images by class in a batch")
        
            train_ds = balanced_image_dataset_from_directory(  # https://keras.io/api/preprocessing/image/
                os.path.join(my_dir_split, "train_augmentation"),
                # num_classes_per_batch must be less than number of available classes in the dataset
                num_classes_per_batch=my_num_classes_per_batch,
                # a specific number of images is selected from every choosen class as long as there are enough samples from this class
                num_images_per_class=my_num_images_per_class,
                labels='inferred',
                label_mode='int',  # means that the labels are encoded as integers
                class_names=None,
                color_mode='rgb',
                image_size=(my_size, my_size),
                shuffle=True,
                # permit to have something random but controled (https://www.w3schools.com/python/ref_random_seed.asp)
                seed=777,
                validation_split=0,
                # safe_triplet=True : does not guarantee that every epoch will include all different samples from the dataset.
                # as sampling is weighted per class, every epoch will include a very high percentage of the dataset and should approach 100%
                # as dataset size increases. This however guarantee that both num_classes_per_batch and num_images_per_class are fixed for all
                # batches including later ones.
                safe_triplet=True,
                samples_per_epoch=None,
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False
            )
        
            # Verification :
            class_names = train_ds.class_names
            print("\nHere the names of the class in the train dataset : ", class_names)
            num_classes = len(class_names)
            print("There are ", num_classes, " classes")
            print("Here the shape of the train dataset :")
            print(train_ds)  # (None, 224, 224, 3) = ? classes, 224x224 pixels (3 : RGB)
        
            # Visualisation :
            print("\nThere are ", len(train_ds), " batches")
            print("Here the labels/classes for each batch : (num_classes_per_batch = ",
                  min_images_par_class, ")")
            for images, labels in train_ds:
                print(labels)
        
        
            # -- 3) The validation dataset val_ds -----------------------------
        
            val_ds = balanced_image_dataset_from_directory(
                os.path.join(my_dir_split, "validation"),
                num_classes_per_batch=my_num_classes_per_batch,
                num_images_per_class=my_num_images_per_class,
                labels='inferred',
                label_mode='int',
                class_names=None,
                color_mode='rgb',
                image_size=(my_size, my_size),
                shuffle=True,
                seed=777,
                validation_split=0,
                safe_triplet=True,
                samples_per_epoch=None,
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False
            )
        
            # Verification
            class_names = val_ds.class_names
            print("\n Here the names of the class in the validation dataset : ", class_names)
            num_classes = len(class_names)
            print("There are ", num_classes, " classes")
            print("Here the shape of the train dataset :")
            print(val_ds)  # (None, 224, 224, 3) = ? classes, 224x224 pixels (3 : RGB)
        
            # Visualisation
            print("\n There are ", len(val_ds), " batches")
            print("Here the labels/classes for each batch : (num_classes_per_batch = ",
                  min_images_par_class, ")")
            for images, labels in val_ds:
                print(labels)
        
            
        
            # ======= II/ THE MODEL CONSTRUCTION ==============================
            # -- 1) The pre-model ---------------------------------------------
        
        
            # (https://inside-machinelearning.com/vgg16-tutoriel-simple-et-detaille/)
        
            # Transfert learning here :
            # - We charge the Keras model to automatically charge the pre-trained weights.
            # - And so we can directly use that neural network
            
            # We test 3 pre-models : EfficientNet, ResNet and DenseNet
            
            for my_premodel_type in my_premodel_type_list:
                
                print("The pre-model type is ", my_premodel_type)
                
                # https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/ResNet152
                if my_premodel_type =="ResNet152":
                    
                    premodel = tf.keras.applications.resnet.ResNet152(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3)
                        )
                    
                # https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet201
                if my_premodel_type =="DenseNet201":
                    
                    premodel = tf.keras.applications.densenet.DenseNet201(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3)
                        )
                
                if my_premodel_type =="EfficientNet":
            
                    if my_size == 224:
                        premodel = tf.keras.applications.EfficientNetB0(
                            weights ='imagenet',
                            include_top = False,
                            input_shape = (224, 224, 3)
                        )
                    elif my_size == 240:
                        premodel = tf.keras.applications.EfficientNetB1(
                            weights ='imagenet',
                            include_top = False,
                            input_shape = (240, 240, 3)
                        )
                    elif my_size == 260:
                        premodel = tf.keras.applications.EfficientNetB2(
                            weights ='imagenet',
                            include_top = False,
                            input_shape = (260, 260, 3)
                        )
                    elif my_size == 300:
                        premodel = tf.keras.applications.EfficientNetB3(
                            weights ='imagenet',
                            include_top = False,
                            input_shape = (300, 300, 3)
                        )
                    elif my_size == 380:
                        premodel = tf.keras.applications.EfficientNetB4(
                            weights ='imagenet',
                            include_top = False,
                            input_shape = (380, 380, 3)
                        )
                    elif my_size == 456:
                        premodel = tf.keras.applications.EfficientNetB5(
                            weights ='imagenet',
                            include_top = False,
                            input_shape = (456, 456, 3)
                        )
                    elif my_size == 528:
                        premodel = tf.keras.applications.EfficientNetB6(
                            weights ='imagenet',
                            include_top = False,
                            input_shape = (528, 528, 3)
                        )
                    elif my_size == 600:
                        premodel = tf.keras.applications.EfficientNetB7(
                            weights ='imagenet',
                            include_top = False,
                            input_shape = (600, 600, 3)
                        )
                
                premodel.summary()
            
                # If we want to search for the best model, we need to test different parameters (my_triplet_choice_list)
                for my_triplet_choice in my_triplet_choice_list:
                    print("----- The metric of the Triplet Loss is ",my_triplet_choice,"-----------")
                    print("The size of my picture is ",my_size)
                    print("There are ",my_num_classes_per_batch," classes by batch")
                    print("There are ",my_num_images_per_class," images by class in a batch")
                    print("The type of my pre-model is ",my_premodel_type)
            
                    
                    for k in range(len(my_embedding_size_list)):
                        
                        my_embedding_size = my_embedding_size_list[k]
                        my_first_dense_size = my_first_dense_size_list[k]
                        print("The embedding size is ",my_embedding_size)
                        print("The first dense size is ",my_first_dense_size)
                
                        # -- 2) The model and its history -------------------------
                
                        model, history = construct_model(premodel, my_embedding_size, my_triplet_choice, my_epoch_number,train_ds,val_ds,my_first_dense_size)
                
                
                        # -- 3) Save the model ------------------------------------
                    
                        
                        # Here the directory to save the model :
                        # - either we want different size layers
                        # - or we want to choose different premodels / size ect
                        # - or the compositions of the batchs
                        if len(my_embedding_size_list) > 1 :
                            # embedding : size of the second dense
                            # firstdense : size of the first dense
                            save_num_classes_images_triplet_choice = str(my_size) + "size_"  + str(my_triplet_choice) + "_firstdense" + str(my_first_dense_size) + "_embedding" + str(my_embedding_size) + "_" + str(my_premodel_type)
                        if len(my_num_classes_per_batch_list) > 1 :
                            # NCB : number of class by batch
                            # NIC : number of image by class
                            save_num_classes_images_triplet_choice = str(my_size) + "size_"  + str(my_triplet_choice) + "_NCB" + str(my_num_classes_per_batch) + "_NIC" + str(my_num_images_per_class) + "_" + str(my_premodel_type)
                        if len(my_num_classes_per_batch_list) ==1 and len(my_embedding_size_list)==1:
                            save_num_classes_images_triplet_choice = str(my_size) + "size_" + str(my_triplet_choice) + "_" + str(my_premodel_type)
                        my_dir_model_eval = os.path.join(my_dir_model,save_num_classes_images_triplet_choice)
                        # We create the folder
                        os.makedirs(my_dir_model_eval)
                        
                        
                        print(tensorflow.__version__)
                        # https://discuss.tensorflow.org/t/using-efficientnetb0-and-save-model-will-result-unable-to-serialize-2-0896919-2-1128857-2-1081853-to-json-unrecognized-type-class-tensorflow-python-framework-ops-eagertensor/12518/10
                        # Here the tensorflow version is 2.11.0
                        # But this version doesn't work to save EfficientNet model
                        # "TypeError: Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON. Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>."
                        # Downgrading from 2.10 to 2.9 solved the issue
                    
                        # https://stackoverflow.com/questions/65085962/downgrading-or-installing-earlier-version-of-a-package-with-pip
                        # On the prompt
                        # To check the packages versions : pip freeze
                        # To downgrade : pip install --upgrade tensorflow==2.9.1
                    
                        # https://www.tensorflow.org/tutorials/keras/save_and_load?hl=fr
                        # We can then save the model
                        model.save(my_dir_model_eval)
                        
                        # ======= III/ EVALUATION HELPING FOR THE MODEL CONSTRUCTION ================
                        # -- 1) Intern evaluation by plotting history ---------------------------------
                    
                        print("The training loss are ", history.history['loss'])
                        print("The validation loss are ", history.history['val_loss'])
                        # We name the folder depending of the model caracteristics
                        my_dir_history_plot_eval = os.path.join(my_dir_history_plot,save_num_classes_images_triplet_choice)
                        # We create the folder
                        os.makedirs(my_dir_history_plot_eval)
                        # We save the plot
                        plot_training(history, my_embedding_size, my_dir_history_plot_eval)
                
                
        # -- 2) Intern evaluation on the premodels by mapping -----------------
        # https://github.com/philipperemy/keract
        
        # If the user want to do the mapping:
        if my_mapping_or_not == "yes":
            
            # We create the folder
            os.makedirs(my_dir_mapping)
            
            # We prepare the pictures : 
            image = Image.open(my_path_image_test) # Open the picture
            image = image.crop((0, 0, my_size, my_size)) # Resize ce picture
            image = img_to_array(image) # Convert the picture in array
            arr_image = np.array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # Reshape the pictures to be as the input of the model to check
            image = preprocess_input(image)
        
            # activations = keract.get_activations(premodel, image, layer_names='block1_conv2') # block1_conv2 or block1_conv1
            activations = keract.get_activations(premodel, image)
        
        
            # We have heat and activation maps :
            keract.display_heatmaps(activations, arr_image, save=True,
                                    directory=os.path.join(my_dir_mapping, "heatmaps"))
            keract.display_activations(activations, cmap=None, save=True, directory=os.path.join(
                my_dir_mapping, "activations"), data_format='channels_last', fig_size=(4, 4), reshape_1d_layers=False)




# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# =============================================================================
# ====== CHOOSE THE VARIABLES =================================================
# =============================================================================

"""
For several variables, you can chose 1 value or several. If you want to construct 1 model, you choose
only 1 value. If you want to construct differents models with theses differents parameters, you choose
the values you want to test. The goal is, thanks to the test script, to choose the best model
"""

# SIZE : size of the pictures
# For testing one model :
ALL_SIZE = [224]
# For testing different models :
#ALL_SIZE = [224, 240, 260, 300, 380, 456, 528, 600]

# NUM_CLASSES_PER_BATCH : number of class by batch
# NUM_IMAGES_PER_CLASS : number of image by class by batch
# If you want to choose a single model :
NUM_CLASSES_PER_BATCH = [2]
NUM_IMAGES_PER_CLASS = [2]
#NUM_CLASSES_PER_BATCH = [30,40,50,30,40,50,12,24,48,20,30,40,50]
#NUM_IMAGES_PER_CLASS = [3,3,3,4,4,4,10,10,10,6,6,6,6]
# If you want to test different models
#NUM_CLASSES_PER_BATCH = [3,10,15,6,3,20,10,30,15]
#NUM_IMAGES_PER_CLASS = [10,3,2,10,20,3,6,2,4]

# Here we are extracting128-dimensional embedding vectors.
# If you want to choose a single model :
EMBEDDING_SIZE = [128]
FIRST_DENSE = [2048]
# If you want to test different models
# EMBEDDING_SIZE = [64,64,64,128,128,128]
# FIRST_DENSE = [2048,1024,512,2048,1024,512]

NB_EPOCH = 2

# TRIPLET_CHOICE : choice of the metric of the triplet loss function
# - If you want to choose a single model :
TRIPLET_CHOICE = ["squared-L2"]  
# - If you want to test different model by modifying triplet distance :
#TRIPLET_CHOICE = ["L2","squared-L2","angular"]

# MAPPING_OR_NOT = if you want to do the heatmaps and activation maps to analyze 
# the model constructed
MAPPING_OR_NOT = "no"

# PREMODEL_CHOSEN
# - Here we put the different pre-models
# /!\ For ResNet152 et DenseNet201 we need to put ALL_SIZE = 224
# /!\ For the different EfficientNet model, the ALL_SIZE depend of the model
# PREMODEL = ["EfficientNet","ResNet152","DenseNet201"]
PREMODEL_CHOSEN = ["EfficientNet"]

# =============================================================================
# ====== CHOOSE THE DIRECTORIES ===============================================
# =============================================================================

# MAIN DIRECTORY : Where we have everything except the code
dir_main = 'D:/deep-learning_re-id_gimenez'
#dir_main_data = "/home/sdb1"
#dir_main_save = "/home/data/marie"
dir_main_data = dir_main
dir_main_save = dir_main
#print("The main directory is ",dir_main,"\n")

# FOLDER SPLIT : Folder with split images which will be used for the deep-learning
# One number for 1 picture size (n*n)
# dataset_split_224 / dataset_split_240 / ... / dataset_split_300
dir_split_all = []
for size_chosen in ALL_SIZE:
    folder_animal_split = 'dataset_split_' + str(size_chosen)
    dir_split = os.path.join(dir_main_data,"1_Pre-processingTEST", folder_animal_split)
    dir_split_all = dir_split_all + [dir_split]
print("The directory of the splitted pictures is ",dir_split_all,"\n")

# FOLDER DATA AUGMENTATION : Data augmentation of the pictures :
#folder_augmentation = "train_augmentation"
folder_augmentation = "train_augmentation"
dir_augmentation = os.path.join(dir_split, folder_augmentation)
print("The directory of the data augmentation pictures is ", dir_augmentation, "\n")

# FOLDER MODEL CONSTRUCTION : where the model, its history and mapping will be saved
dir_construction = os.path.join(dir_main_save,"2_Model-constructionTEST")
# The model directory :
dir_model = os.path.join(dir_construction,"result_model")
# The mapping directory and its example picture
path_image_test = os.path.join(dir_construction,"Isis_resize300.jpg")
dir_mapping = os.path.join(dir_construction,"result_mapping_efficientnetB3")
# The history mapping
dir_history_plot = os.path.join(dir_construction,"result_plot_loss")


# =============================================================================
# ====== LAUNCH THE SCRIPT ====================================================
# =============================================================================

modelconstruction(my_dir_augmentation=dir_augmentation,my_dir_split_list=dir_split_all,my_size_list=ALL_SIZE,my_num_classes_per_batch_list=NUM_CLASSES_PER_BATCH,my_num_images_per_class_list=NUM_IMAGES_PER_CLASS,my_embedding_size_list=EMBEDDING_SIZE,my_triplet_choice_list=TRIPLET_CHOICE,my_epoch_number=NB_EPOCH,my_path_image_test=path_image_test,my_dir_history_plot = dir_history_plot,my_dir_model=dir_model,my_dir_mapping=dir_mapping,my_mapping_or_not=MAPPING_OR_NOT,my_folder_augmentation=folder_augmentation,my_premodel_type_list=PREMODEL_CHOSEN,my_first_dense_size_list=FIRST_DENSE)

