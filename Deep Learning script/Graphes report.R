# TITLE : PROJET DEEP LEARNING LYNX - Stage M2 2023

# Author : Marie Joigneau
# Supervisor : Olivier Gimenez

# Problematic : Re-identify individual from eurasian lynx population

# This part : Here we do all the graphes needed for the analysis & and the presentation
# As the presentation is in French, the titles and legends are written in French
# However, the explanation remains in English to be understood

# =============================================================================
# =============================================================================
# ======================== PACKAGES ===========================================
# =============================================================================
# =============================================================================


library(ggplot2)
library(tibble)
library(readr)
library(reshape2)
library(wesanderson)
library(ggpubr)


# =============================================================================
# =============================================================================
# =========================== MAIN ============================================
# =============================================================================
# =============================================================================


# ===============================================================================
# ============== I/ HISTOGRAM PICTURE BY Catégorie (GLOBAL DATASET) =============
# ===============================================================================

# EXPLAINATION :
# Here we want to obtain an histogram of the number of pictures by individual and by category


# ========= 1) We create the dataframe ==========================================

# --- A) Number of pictures (files) for each individual and category ------------

# https://stackoverflow.com/questions/60755023/r-how-to-count-the-number-of-files-in-each-folder

# We write the folder names :
folder_OCS <- "D:/deep-learning_re-id_gimenez/0_dataset_Marie/0_dataset_Marie_OCS"
folder_OFB_ocelles <- "D:/deep-learning_re-id_gimenez/0_dataset_Marie/0_dataset_Marie_OFB_ocelles"
folder_OFB_spots <- "D:/deep-learning_re-id_gimenez/0_dataset_Marie/0_dataset_Marie_OFB_spots"

# Then we obtain a list of all the files in each folder :
files_OCS <-  list.files(folder_OCS, pattern = ".", all.files = FALSE, recursive = TRUE, full.names = TRUE)
files_OFB_ocelles <-  list.files(folder_OFB_ocelles, pattern = ".", all.files = FALSE, recursive = TRUE, full.names = TRUE)
files_OFB_spots <-  list.files(folder_OFB_spots, pattern = ".", all.files = FALSE, recursive = TRUE, full.names = TRUE)

# And we split the files by the folder they belong
dir_list_OCS <- split(files_OCS, dirname(files_OCS))
dir_list_OFB_ocelles <- split(files_OFB_ocelles, dirname(files_OFB_ocelles))
dir_list_OFB_spots <- split(files_OFB_spots, dirname(files_OFB_spots))

# And we finally obtain the number of files in each subfolder = number of pictures by individual
Number_of_pictures_OCS <- sapply(dir_list_OCS, length)
Number_of_pictures_OFB_ocelles <- sapply(dir_list_OFB_ocelles, length)
Number_of_pictures_OFB_spots <- sapply(dir_list_OFB_spots, length)


# --- B) Dataframe : number of pictures for each individual ---------------------

# We convert the list into dataframe
df_OCS = as.data.frame(Number_of_pictures_OCS)
df_OFB_ocelles = as.data.frame(Number_of_pictures_OFB_ocelles)
df_OFB_spots = as.data.frame(Number_of_pictures_OFB_spots)

# We rename the columns :
names(df_OCS)[1] <- "Nombre"
names(df_OFB_ocelles)[1] <- "Nombre"
names(df_OFB_spots)[1] <- "Nombre"
df_OCS$Origine = "Pas d'information"
df_OFB_ocelles$Origine = "Ocelles"
df_OFB_spots$Origine = "Spots"

# We merge all the 3 dataframes to obtain the main dataframe
df = rbind(df_OCS,df_OFB_ocelles,df_OFB_spots)


# ========= 2) We create the histogram ==========================================


# Here we have the histogram of the number of pictures by individual and by category
ggplot(df, aes(x=Nombre, color=Origine)) +
  geom_histogram(fill="white") +
  xlab("Nombre d'images") +
  ylab("Nombre d'individus") + 
  labs(title="Distribution empilée des images par origine") + 
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(plot.title = element_text(hjust = 0.5))

# And we save it in a safe place
ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/1_distribution_empilee_images_origines.png")



# ===============================================================================
# ============== II/ HISTOGRAM PICTURE BY INDIVIDUAL (TEST DATASET) =============
# ===============================================================================

# EXPLAINATION :
# Here we want to obtain an histogram of the number of pictures by individual in the test dataset


# ========= 1) We create the dataframe ==========================================

# --- A) Number of pictures (files) for each individual -------------------------

# https://stackoverflow.com/questions/60755023/r-how-to-count-the-number-of-files-in-each-folder

# We write the folder names :
folder_test <- "D:/deep-learning_re-id_gimenez/1_Pre-processing/dataset_split_224/test_1-photo-supprimee"
# Then we obtain a list of all the files in each folder :
files_test <-  list.files(folder_test, pattern = ".", all.files = FALSE, recursive = TRUE, full.names = TRUE)

# And we split the files by the folder they belong
dir_list_test <- split(files_test, dirname(files_test))
# And we finally obtain the number of files in each subfolder = number of pictures by individual
Number_of_pictures_test <- sapply(dir_list_test, length)

# --- B) Dataframe : number of pictures for each individual ---------------------

# We convert the list into dataframe
df_test = as.data.frame(Number_of_pictures_test)
# We rename the column :
names(df_test)[1] <- "Nombre"


# ========= 2) We create the histogram ==========================================

ggplot(df_test, aes(x=Nombre)) +
  geom_histogram(alpha=0.3, col="black",fill="red",binwidth = 1)+ 
  scale_fill_gradient(low = "black", high = "white")+
  xlab("Nombre d'images") +
  ylab("Nombre d'individus") + 
  labs(title="Distribution des images par individu du jeu de données test") + 
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(plot.title = element_text(hjust = 0.5))

# And we save the graphe
ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/1_distribution_empilee_images_dataset_test.png")



# ===============================================================================
# ============== III/ BOXPLOT COMPARISON TRIPLET LOSS METRICS ===================
# ===============================================================================

# EXPLAINATION :
# Here we want to obtain boxplots which compare the different models based on the Triplet Loss metrics

# We open the dataframe :
df <- read.csv("D:/OneDrive/Stage M2 - Montpellier Gimenez/PARTIE 1 - deep learning/code/results_test.csv",header=TRUE,sep=";",dec=".",fileEncoding="utf-8")

# We don't need theses columns
df$Modele <- NULL
df$eval <- NULL

# We want to merge the columns to be able to do the boxplot correctly
mdf <- melt(df, id.vars="TL")
names(mdf)[1] <- "Triplet_Loss"

# We want to use wesanderson palette colors :
pal <- wes_palette("Moonrise3", n = 3, type = "discrete")

ggplot(mdf, aes(x = variable, y = value, fill = Triplet_Loss)) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  scale_fill_manual(values = pal) +
  theme(legend.position = "top")+
  xlab("Métrique d'évaluation") +
  ylab("Valeur") + 
  labs(title="Performance des différents modèles") + 
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/3_boxplot_performance_par_TL.png")



# ===============================================================================
# ============== IV/ GEOM_BAR COMPARISON PRE-MODELS =============================
# ===============================================================================

# EXPLAINATION :
# Here we want to obtain an geom_bar which compare the different models based on the pre-model

# We open the dataframe :
df <- read.csv("D:/OneDrive/Stage M2 - Montpellier Gimenez/PARTIE 1 - deep learning/code/results_test.csv",header=TRUE,sep=";",dec=".",fileEncoding="utf-8")

# We only take the best combination already chosen of the Triplet Loss metric 
df <- df[df$TL=="L2-carre",]
df$TL <- NULL
# ... and k-neighboors metric
df <- df[df$eval=="Angulaire",]
df$eval <- NULL

# We want to merge the columns to be able to do the geom_bar correctly
mdf <- melt(df, id.vars="Modele")
# And we round the values to be readable
mdf$value <- round(mdf$value,2)

# And we rename the column
names(mdf)[1] <- "Prémodèle"

# We want to use wesanderson palette colors :
pal <- wes_palette("Moonrise3", n = 3, type = "discrete")

ggplot(mdf, aes(variable, value, fill=Prémodèle)) + 
  geom_bar(stat="identity", position="dodge",alpha=0.8,color="black") +
  scale_fill_manual(values = pal) +
  geom_text(aes(label=value), vjust=0,position = position_dodge(width = 1)) +
  coord_cartesian(ylim = c(0.42, 0.673))+
  xlab("Métrique d'évaluation") +
  ylab("Valeur") + 
  labs(title="Performance des différents modèles") + 
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(plot.title = element_text(hjust = 0.5))

# And we save the graphe
ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/4_geom_bar_performance_par_pre-model.png")


# ===============================================================================
# ============== V/ GEOM_LINES COMPARAISON TRESHOLD  ============================
# ===============================================================================

# EXPLAINATION :
# Here we want to obtain geom_lines to choose the best treshold depending of true positive and true negative results by treshold

# We open the csv :
df <- read.csv("D:/OneDrive/Stage M2 - Montpellier Gimenez/PARTIE 1 - deep learning/code/results_treshold.csv",header=TRUE,sep=";",dec=".",fileEncoding="utf-8")

# We rename the columns :
names(df)[2] <- "Vrais positifs (TP)"
names(df)[3] <- "Vrais négatifs (TN)"

# We don't want the TP_folder column :
df$TP_folder <- NULL

# We merge the dataframe to be able to to the geom_line correctly :
mdf <- melt(df, id.vars="treshold")

# We put it in % :
mdf$value = 100 * mdf$value
# We rename the columns :
names(mdf)[2] <- "Métrique"

# We want to use wesanderson palette colors :
pal <- wes_palette("Moonrise3", n = 3, type = "discrete")

# Modele avec smooth :
ggplot(mdf, aes(x = treshold, y = value, group = Métrique)) +
  geom_smooth(aes(color = Métrique), method = "gam", size = 1.3) +
  scale_color_manual(values = pal) +  # Utiliser scale_color_manual() pour les couleurs des lignes
  theme(legend.position = "top") +
  xlab("Seuil") +
  ylab("Performance (en %)") +
  labs(title = "Performance du modèle avec différents seuils") +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(plot.title = element_text(hjust = 0.5))


# And we save the graphe
ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/5_geom_smooth_meilleur_treshold.png")



# ===============================================================================
# ============== VI/ DISTRIBUTION OF PREDICTED GROUPS ===========================
# ===============================================================================

# EXPLAINATION :
# Here we want to analyze the distribution of the groups of the individuals predicted
# We have :
# - Alone and bad classed individuals
# - Well classed individuals
# - Bad classed individuals

# ========= 1) Distribution for the SCR =========================================

# We open the csv
df <- read.csv("D:/OneDrive/Stage M2 - Montpellier Gimenez/PARTIE 1 - deep learning/code/results_distributionSCR.csv",header=TRUE,sep=";",dec=".",fileEncoding="utf-8")

# We create the 1st dataframe with the count of the alone and not well classed individuals
df1 = df[(df$well_classed=="no"),]
len_alone_new = length(df1[df1$alone_and_new=="yes",]$nb)
df1 = data.frame(Var1=1,well_classed="alone",Freq=len_alone_new)

# We create the dataframe with the well classed individuals
df2 = df[df$well_classed=="yes",]
df2 = data.frame(table(df2$nb))
df2$well_classed = "yes"

# We create the dataframe with the no well classed individuals in groups
df3 = df[(df$well_classed=="no"),]
df3 = df3[(df3$alone_and_new=="no"),]
df3 = data.frame(table(df3$nb))
df3$well_classed = "no"

# We merge theses 3 dataframes
df_histo = rbind(df1,df2,df3)
# We convert the values into integer
df_histo$Var1 <- as.integer(df_histo$Var1)

# We do some esthetics things for the ggplot :
names(df_histo)[2] <- "Prédiction"
# We rename in french for the presentation :
df_histo[(df_histo$Prédiction == "alone"),]$Prédiction <- "Seul et mal classé"
df_histo[(df_histo$Prédiction == "no"),]$Prédiction <- "Mal classé"
df_histo[(df_histo$Prédiction == "yes"),]$Prédiction <- "Bien classé"

# We want to use wesanderson palette colors :
pal <- wes_palette("Moonrise3", n = 3, type = "discrete")

histoSCR <- ggplot(df_histo, aes(Var1, Freq, fill=Prédiction)) + 
  geom_bar(stat="identity", position="dodge",alpha=0.8,color="black") +
  scale_fill_manual(values = pal) +
  xlab("Nombre d'images dans le groupe") +
  ylab("") +  
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(plot.title = element_text(hjust = 0.5))

# And we save the graphe
ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/6_geom_bar_distribution_prediction_SCR.png")


# ========= 2) Distribution for the train dataset ===============================

# We open the csv
df <- read.csv("D:/OneDrive/Stage M2 - Montpellier Gimenez/PARTIE 1 - deep learning/code/results_distributionTRAIN.csv",header=TRUE,sep=";",dec=".",fileEncoding="utf-8")

# We create the 1st dataframe with the count of the alone and not well classed individuals
df1 = df[(df$well_classed=="no"),]
len_alone_new = length(df1[df1$alone_and_new=="yes",]$nb)
df1 = data.frame(Var1=1,well_classed="alone",Freq=len_alone_new)

# We create the dataframe with the well classed individuals
df2 = df[df$well_classed=="yes",]
df2 = data.frame(table(df2$nb))
df2$well_classed = "yes"

# We create the dataframe with the no well classed individuals in groups
df3 = df[(df$well_classed=="no"),]
df3 = df3[(df3$alone_and_new=="no"),]
df3 = data.frame(table(df3$nb))
df3$well_classed = "no"

# We merge theses 3 dataframes
df_histo = rbind(df1,df2,df3)
# We convert the values into integer
df_histo$Var1 <- as.integer(df_histo$Var1)

# We do some esthetics things for the ggplot :
names(df_histo)[2] <- "Prédiction"
# We rename in french for the presentation :
df_histo[(df_histo$Prédiction == "alone"),]$Prédiction <- "Seul et mal classé"
df_histo[(df_histo$Prédiction == "no"),]$Prédiction <- "Mal classé"
df_histo[(df_histo$Prédiction == "yes"),]$Prédiction <- "Bien classé"

# We want to use wesanderson palette colors :
pal <- wes_palette("Moonrise3", n = 3, type = "discrete")

histoTRAIN <-ggplot(df_histo, aes(Var1, Freq, fill=Prédiction)) + 
  geom_bar(stat="identity", position="dodge",alpha=0.8,color="black") +
  scale_fill_manual(values = pal) +
  xlab("Nombre d'images dans le groupe") +
  ylab("Nombre de de groupes") +  
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(plot.title = element_text(hjust = 0.5))

# And we save the graphe
ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/6_geom_bar_distribution_prediction_TRAIN.png")


# ========= 3) Merge of the 2 graphs ============================================

# We merge
plot <- ggarrange(histoTRAIN, histoSCR, ncol=2, nrow=1, common.legend = TRUE, legend="bottom", widths = c(0.65, 0.35)) 
# And we put a title
plot <- annotate_figure(plot, top = text_grob("Distribution des individus prédits TRAIN / SCR", 
                                      color = "black", size = 14))
plot

# And we save the graphe
ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/5-6_geom_bar_distribution_prediction_THE2.png")



# ===============================================================================
# ============== VII/ RESULTS OF THE SCR MODELS =================================
# ===============================================================================

df <- read.csv("D:/OneDrive/Stage M2 - Montpellier Gimenez/PARTIE 2 - spatial capture recapture/SCR marie/results_model2.csv",header=TRUE,sep=";",dec=".",fileEncoding="utf-8")

df$values <- as.numeric(df$values)
names(df)[4] <- "Catégorie"
df[df$Catégorie == "IA",]$Catégorie <- "Intelligence Artificielle"

# We put back the right values in negative
# df[df$metrique == "P0",]$values <- - df[df$metrique == "P0",]$values
# df[df$metrique == "D0",]$values <- - df[df$metrique == "D0",]$values

# We want to use wesanderson palette colors :
pal <- wes_palette("Moonrise3", n = 3, type = "discrete")


dfSIGMA <- df[df$metrique == "Sigma",]
pointSIGMA <- ggplot(dfSIGMA, aes(metrique, values, color=Catégorie, y=values, ymin=values-IC, ymax=values+IC)) + 
  geom_pointrange(position = position_dodge(width = 0.2)) +  # Ajout du décalage des points
  scale_color_manual(values = wes_palette("Moonrise3", n = 3, type = "discrete")) +  # Utilisation de scale_color_manual
  xlab("") +
  ylab("") +  
  theme_bw() + 
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    plot.title = element_text(hjust = 0.5)
  )

dfP0 <- df[df$metrique == "P0",]
pointP0 <- ggplot(dfP0, aes(metrique, values, color=Catégorie, y=values, ymin=values-IC, ymax=values+IC)) + 
  geom_pointrange(position = position_dodge(width = 0.2)) +  # Ajout du décalage des points
  scale_color_manual(values = wes_palette("Moonrise3", n = 3, type = "discrete")) +  # Utilisation de scale_color_manual
  xlab("") +
  ylab("Valeur") +  
  theme_bw() + 
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    plot.title = element_text(hjust = 0.5)
  )


dfD0 <- df[df$metrique == "D0",]
pointD0 <- ggplot(dfD0, aes(metrique, values, color=Catégorie, y=values, ymin=values-IC, ymax=values+IC)) + 
  geom_pointrange(position = position_dodge(width = 0.2)) +  # Ajout du décalage des points
  scale_color_manual(values = wes_palette("Moonrise3", n = 3, type = "discrete")) +  # Utilisation de scale_color_manual
  xlab("") +
  ylab("") +  
  theme_bw() + 
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    plot.title = element_text(hjust = 0.5)
  )

dfN <- df[df$metrique == "N",]
pointN <- ggplot(dfN, aes(metrique, values, color=Catégorie, y=values, ymin=values-IC, ymax=values+IC)) + 
  geom_pointrange(position = position_dodge(width = 0.2)) +  # Ajout du décalage des points
  scale_color_manual(values = wes_palette("Moonrise3", n = 3, type = "discrete")) +  # Utilisation de scale_color_manual
  xlab("") +
  ylab("") +  
  theme_bw() + 
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    plot.title = element_text(hjust = 0.5)
  )

dfDENS <- df[df$metrique == "Densite",]
pointDENS <- ggplot(dfDENS, aes(metrique, values, color=Catégorie, y=values, ymin=values-IC, ymax=values+IC)) + 
  geom_pointrange(position = position_dodge(width = 0.2)) +  # Ajout du décalage des points
  scale_color_manual(values = wes_palette("Moonrise3", n = 3, type = "discrete")) +  # Utilisation de scale_color_manual
  ylim(1,17)+
  xlab("") +
  ylab("") +  
  theme_bw() + 
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    plot.title = element_text(hjust = 0.5)
  )

# We merge
plot <- ggarrange(pointP0, pointSIGMA, pointD0, pointN, pointDENS, ncol=5, nrow=1, common.legend = TRUE, legend="top") 
# And we put a title
plot <- annotate_figure(plot, top = text_grob("Résultats du modèle par l'humain et l'IA", 
                                              color = "black", size = 14),bottom = text_grob("Paramètre du modèle", 
                                                                                          color = "black", size = 10))
plot

# And we save the graphe
ggsave(filename = "D:/OneDrive/Stage M2 - Montpellier Gimenez/memoire/graphes/7_resultats_modele_SCR_IA_humain.png")




