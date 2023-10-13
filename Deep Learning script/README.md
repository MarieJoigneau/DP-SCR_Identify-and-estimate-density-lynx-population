# DP-SCR_Identify-and-estimate-density-lynx-population
Script and configuration


Here you have all the script I use to do my deep learning model and predictions.

You also have the skeleton of my folders to understand how I have done. 
As you work with OFB, you will also have my pictures.

I advise you to read my report and my presentation before the script for a better understanding.

For the script : 
- 1_Pre-processing : permits to prepare the dataset for the model (find the individual on the picture, crop, resize and split the dataset in train/validation/test)
- 1_Pre-processing_change-min-picture : permits to keep the same distribution but delete the individuals with less than X pictures
- 2_Model-construction : permits to create different models depending of Triplet Loss and KNN metrics and the tranfer learning model
- 2_Model-construction_batch-composition : permits to create different models depending of the batch composition (number of class by batch and number of pictures by batch)
- 2_Model-construction_data-augmentation : permits to create another model without data augmentation
- 2_Model-construction_embedding : permits to create different models depending of the sizes of the dense layers
- 2_Model-construction_min-pic-15 : permits to create another model with at least 15 pictures by individual in the dataset
- 2_Model-construction_BEST-MODEL : I have copied the best model and renamed it with its characteristics for you to remind them
- 3_Model-test : permits to test the different models to choose the best
- 4_Best-treshold : permits to choose the best treshold when you want to differenciate a new from a re identification (we are in an open population)
- 5_Model-prediction-already-dataset_automatique : predicts new individuals (in 0new folder) when you already have a dataset annoted (automatic way)
- 5_Model-prediction-already-dataset_manuel : predicts new individuals (in 0new folder) when you already have a dataset annoted (manual way)
- 6_Model-prediction_new-dataset : predicts a whole new dataset
- 7_Examine-distribution : examines the distribution of the individuals after their prediction
- 8_individuals-corresponding-for-SCR : gives the association folder-pictures to compare with the human prediction (for Spatial Capture Recapture part)
- Graphes report : all the graphes I have done in my internship

For the research of the best model, I have done the following order : 
TL / KNN metrics 
THEN transfer learning models 
THEN min pic 15 
THEN embedding
THEN batch composition
THEN data augmentation

Don't hesitate to contact me if you have any question,
Best,
Marie Joigneau