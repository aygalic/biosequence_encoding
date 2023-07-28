library(mvtnorm)
library(rgl)
library(MASS)
library("readxl")
library(Rtsne)
library(ggplot2)
library(ggpubr)
library("purrr")
library("viridis")   
library(plotly)
library(scales)

setwd("~/Thesis/genome_analysis_parkinson/src")

table = fread("../workfiles/processed_data.csv", header = T)

# these are the file names for each encoded observation
names = table$name

# this is the corresponding encoding
encoded_experession = table[,1:256] 




#################################
################################# using metadata to build curves
#################################



meta <- read_excel("../../METADATA_200123.xlsx", sheet = "Foglio1")
patient_ids <- sapply(names, function(names) c(strsplit(names, ".", fixed = T)[[1]][2]), USE.NAMES=FALSE)
cohorts = meta$Cohort[match(patient_ids, meta$`Patient Number`)]
#phases <- sapply(names, function(names) c(strsplit(names, "-", fixed = T)[[1]][2]), USE.NAMES=FALSE)
time_points <- sapply(names, function(names) c(strsplit(names, ".", fixed = T)[[1]][3]), USE.NAMES=FALSE)




# what are the unique patient ID ?
unique_ids = unique(patient_ids)


# here 
files_per_id = sapply(unique_ids, function(x) names[which(patient_ids %in% x)]) 

sapply(unique_ids, function(x) names[which(patient_ids %in% x)] ) 


# reorder ou dataset in a way that every ID has a row for a corresponding timepoint

########################
# IDxxxx BL    ...
# IDxxxx V02   ...
# IDxxxx V04   ...
# IDyyyy BL    ...
# IDyyyy V02   ...
# IDyyyy V04   ...
########################




find_files_per_id = function(x){
  # we take all the entires (primary keys) for a given patient name
  obs = names[which(patient_ids %in% x)]
  # we load the encoded data corresponding and we bind the timepoint to it
  data = map(obs, (function(o) data.table(table[name==o,], as.list(strsplit(o, ".", fixed = T)[[1]][3])) ))
  # assemble it into a data.table
  curr_table = rbindlist(data)
  return(curr_table)
}

id_wise_data = map(unique_ids, find_files_per_id)
new_data_table = rbindlist(id_wise_data)



# now we would want to double order it
 
# then we plot a couple of curves 

# then we check if they are random noise or not...
 




