library(mvtnorm)

library(rgl)
library(MASS)

library("readxl")

library(Rtsne)
library(ggplot2)

library(ggpubr)
library("purrr")
library(data.table)



setwd("~/Library/Thesis/genome_analysis_parkinson/src")

# let's work with a somewhat decent dataframe first
table = fread("../workfiles/processed_data.csv", header = T)





names = table$name
names


encoded_experession = table[,1:64] 



#################################
################################# using metadata to build curves
#################################



meta <- read_excel("../../METADATA_200123.xlsx", sheet = "Foglio1")
patient_ids <- sapply(names, function(names) c(strsplit(names, ".", fixed = T)[[1]][2]), USE.NAMES=FALSE)
cohorts = meta$Cohort[match(patient_ids, meta$`Patient Number`)]
#phases <- sapply(names, function(names) c(strsplit(names, "-", fixed = T)[[1]][2]), USE.NAMES=FALSE)
time_points <- sapply(names, function(names) c(strsplit(names, ".", fixed = T)[[1]][3]), USE.NAMES=FALSE)




# what are the unique patient ID ?
length(patient_ids)
length(unique(patient_ids))


 
unique_ids = unique(patient_ids)




lapply(unique_ids, function(x) which(patient_ids %in% x)) 
lapply(unique_ids, function(x) names[which(patient_ids %in% x)]) 

# here 
files_per_id = sapply(unique_ids, function(x) names[which(patient_ids %in% x)]) 

sapply(unique_ids, function(x) names[which(patient_ids %in% x)] ) 




ffffff = function(x){
  obs = names[which(patient_ids %in% x)]
  data = list()
  for(o in obs){
    path = paste0("../../data/quant/",o)
    print(path)
    data = append(data, fread(path))
  }
  return(data)
}

thingee = sapply(unique_ids, ffffff)





 
 




