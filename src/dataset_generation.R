#install.packages("readxl")
library("readxl")
library(MASS)
library(glmnet)
library("purrr")
library("ggplot2")  
library("Rfast")
library("data.table")

# The reasoning behind this file is that:
# Since the files we are dealing with make the total dataset impossible to 
# handle in memory, we are going to build a pipeline that apply a function to
# each files (like average, variances etc....). We will do this
# without loading everything in  memory and just reading from the disk, 
# in order to get some summary statistics about our dataset

setwd("~/Library/CloudStorage/OneDrive-Personal/polimi/Thesis")
meta <- read_excel("METADATA_200123.xlsx", sheet = "Foglio1")


meta$`Patient Number`



#dir <- system.file("./data/quant/", package = "tximportData")
dir <- "./data/quant"
list.files(dir) # up to this everything works

samples <- list.files(dir)
samples

files <- file.path(dir, samples)
#names(files) <- paste0("sample", 1:417)



#files <- list.files(dir)


all(file.exists(files))
length(files)


files


time_points <- sapply(files, function(files) c(strsplit(files, ".", fixed = T)[[1]][4]), USE.NAMES=FALSE)
time_points

file_type <- sapply(files, function(files) c(strsplit(files, ".", fixed = T)[[1]][9]), USE.NAMES=FALSE)
file_type

patient_ids <- sapply(files, function(files) c(strsplit(files, ".", fixed = T)[[1]][3]), USE.NAMES=FALSE)
patient_ids

phases <- sapply(files, function(files) c(strsplit(files, "-", fixed = T)[[1]][2]), USE.NAMES=FALSE)
phases



# now we just wanna link the patient number to the cohort for the sake of a plot
cohorts = meta$Cohort[match(patient_ids, meta$`Patient Number`)]






# this function load all patients in a given files
# it returns a dataset where each line correspon to a patient, and each column
# correspond to a gene
# values correspond to the TPM values.


#files_to_dataframe <- function(files){
#  n = length(files)
#  #print(paste("length :", n))
#  i = 0
#  # initiate dataframe
#  ds = NULL
#  # for each file 
#  for(file in files){
#    i = i + 1
#    #print(paste("current file :",i ,"/",n))
#    # pick one file and load it
#    patient_data = read.delim(file, header = TRUE)[,4] # we only keep TPM
#    # to avoid artifacts
#    if(length(patient_data)==95309){
#      ds <- rbind(ds, patient_data)
#    }
#  }
#  return (ds)
#} 

### This approach has a huge problem,
### Repetitively using rbind is extremely slow and we might seek alternatives 
### to this method
### it gets exponentially slower 




# this function needs the meta file as part of the environement
file_to_row_matrix <- function(file, add_metadata = FALSE){
  # initiate dataframe
  line = NULL
  # pick one file and load it
  patient_data = fread(file, header = TRUE)[,4] # we only keep TPM
  # to avoid artifacts
  if(dim(patient_data)[1]==95309){
    if(add_metadata){
      filename = strsplit(file, "/", fixed = T)[[1]][4]
      clinical_event <- strsplit(filename, ".", fixed = T)[[1]][3]
      phase <- strsplit(filename, "-", fixed = T)[[1]][2]
      patient_id <- strsplit(filename, ".", fixed = T)[[1]][2]
      cohort = meta$Cohort[match(patient_id, meta$`Patient Number`)][1]
      
      line = c(filename,patient_id, clinical_event, phase, cohort, patient_data$TPM)
    }
    else{
      #line = as.matrix(patient_data, nrow = 1 )
      line = patient_data
    }
  }
  return (line)
} 


# this function needs the meta file as part of the environement
file_to_vec <- function(file, add_metadata = FALSE){
  # initiate dataframe
  line = NULL
  # pick one file and load it
  patient_data = fread(file, header = TRUE)[,4] # we only keep TPM
  
  # to avoid artifacts
  if(dim(patient_data)[1]==95309){
    if(add_metadata){
      filename = strsplit(file, "/", fixed = T)[[1]][4]
      clinical_event <- strsplit(filename, ".", fixed = T)[[1]][3]
      phase <- strsplit(filename, "-", fixed = T)[[1]][2]
      patient_id <- strsplit(filename, ".", fixed = T)[[1]][2]
      cohort = meta$Cohort[match(patient_id, meta$`Patient Number`)][1]
      
      line = c(filename,patient_id, clinical_event, phase, cohort, patient_data$TPM)
    }
    else{
      #line = as.matrix(patient_data, nrow = 1 )
      line = patient_data
    }
  }
  return (line)
} 



files_to_dataframe_fast <- function(files, keep_file_names = F){
  n = length(files)
  ds  = map(as.matrix(files, ncol = 1 ), file_to_row_matrix, keep_file_names)
  ds_ = t(as.data.frame(ds, col.names = 1:n))
  return (ds_)
} 


# not actually faster
fasterrrr <- function(files, keep_file_names = F){
  list_of_dfs <- lapply(files, file_to_vec, add_metadata = T)
  rows <- Reduce(rbind,list_of_dfs)
  return (rows)
}

system.time(
  ds_transcript_test_slowwwww  <- files_to_dataframe_fast(files_transcript[1:10], T)
)

system.time(
  ds_transcript_testerrrr  <- fasterrrr(files_transcript[1:10], T)
)






# create a simple csvfile out of all the transcripts with all information
# about cohort, timepoint etc.

files_transcript = files[file_type == "transcripts"]

# remove NA
files_transcript = files_transcript[!is.na(files_transcript)]


# into a dataframe
ds_transcript  = files_to_dataframe_fast(files_transcript, T)

# we trim the filename column and make it into different features of interst
ds_transcript = as.data.frame(ds_transcript)
colnames(ds_transcript)[1:5] = c("file_name", "patient_id", "clinical_event", "phase", "cohort")

# we set the name of the features
genes = read.delim(files[10], header = TRUE)[,1]
colnames(ds_transcript)[6:(length(ds_transcript))] = genes


#write.csv(ds_transcript, "./workfiles/raw_data.csv")
fwrite(ds_transcript, "./workfiles/raw_data.csv")



# normalization step 
# min max normalization can be an issue : with degenerative deseases we are prone
# to face outliers in data
#
# we will use robust scaller but we have to find some litterature about normalization
# choices for this kind of data
robust_scalar<- function(x){(x- median(x)) /(quantile(x,probs = .75)-quantile(x,probs = .25))}

robust_scalar_transcripts <- as.data.frame(lapply(ds_transcript, robust_scalar))








# we first want to compare the transcript expression at diagnosis and after 6 moths
files_BL_1  = files[time_points=="BL" & file_type == "transcripts" & cohorts == 1]
files_V02_1 = files[time_points=="V02"& file_type == "transcripts" & cohorts == 1]
files_V04_1 = files[time_points=="V04"& file_type == "transcripts" & cohorts == 1]
files_V06_1 = files[time_points=="V06"& file_type == "transcripts" & cohorts == 1]
files_V08_1 = files[time_points=="V08"& file_type == "transcripts" & cohorts == 1]

# remove NA
files_BL_1 = files_BL_1[!is.na(files_BL_1)]
files_V02_1 = files_V02_1[!is.na(files_V02_1)]
files_V04_1 = files_V04_1[!is.na(files_V04_1)]
files_V06_1 = files_V06_1[!is.na(files_V06_1)]
files_V08_1 = files_V08_1[!is.na(files_V08_1)]


# get the corresponding dataframes
#ds_BL_1  = files_to_dataframe(files_BL_1)
#ds_V02_1 = files_to_dataframe(files_V02_1)
#ds_V04_1 = files_to_dataframe(files_V04_1)
#ds_V06_1 = files_to_dataframe(files_V06_1)
#ds_V08_1 = files_to_dataframe(files_V08_1)


ds_BL_1  = files_to_dataframe_fast(files_BL_1)
ds_V02_1 = files_to_dataframe_fast(files_V02_1)
ds_V04_1 = files_to_dataframe_fast(files_V04_1)
ds_V06_1 = files_to_dataframe_fast(files_V06_1)
ds_V08_1 = files_to_dataframe_fast(files_V08_1)


saveRDS(ds_BL_1, "workfiles/ds_BL_1.rds")
saveRDS(ds_V02_1, "./workfiles/ds_V02_1.rds")
saveRDS(ds_V04_1, "./workfiles/ds_V04_1.rds")
saveRDS(ds_V06_1, "./workfiles/ds_V06_1.rds")
saveRDS(ds_V06_1, "./workfiles/ds_V08_1.rds")







# same for cohort healthy
# we first want to compare the transcript expression at diagnosis and after 6 moths
files_BL_2  = files[time_points=="BL" & file_type == "transcripts" & cohorts == 2]
files_V02_2 = files[time_points=="V02"& file_type == "transcripts" & cohorts == 2]
files_V04_2 = files[time_points=="V04"& file_type == "transcripts" & cohorts == 2]
files_V06_2 = files[time_points=="V06"& file_type == "transcripts" & cohorts == 2]
files_V08_2 = files[time_points=="V08"& file_type == "transcripts" & cohorts == 2]



# remove NA
files_BL_2 = files_BL_2[!is.na(files_BL_2)]
files_V02_2 = files_V02_2[!is.na(files_V02_2)]
files_V04_2 = files_V04_2[!is.na(files_V04_2)]
files_V06_2 = files_V06_2[!is.na(files_V06_2)]
files_V08_2 = files_V08_2[!is.na(files_V08_2)]

# get the corresponding dataframes:
ds_BL_2  = files_to_dataframe_fast(files_BL_2)
ds_V02_2 = files_to_dataframe_fast(files_V02_2)
ds_V04_2 = files_to_dataframe_fast(files_V04_2)
ds_V06_2 = files_to_dataframe_fast(files_V06_2)
ds_V08_2 = files_to_dataframe_fast(files_V08_2)



saveRDS(ds_BL_2, "./workfiles/ds_BL_2.rds")
saveRDS(ds_V02_2, "./workfiles/ds_V02_2.rds")
saveRDS(ds_V04_2, "./workfiles/ds_V04_2.rds")
saveRDS(ds_V06_2, "./workfiles/ds_V06_2.rds")
saveRDS(ds_V06_2, "./workfiles/ds_V08_2.rds")



ds = rbind(ds_BL_1, ds_BL_2)

saveRDS(ds, "./workfiles/ds_BL_1_and_2.rds")








