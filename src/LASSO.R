#install.packages("readxl")
library("readxl")
library(MASS)
library(glmnet)
library("purrr")
library("ggplot2")  
library("Rfast")
library("data.table")

# In this file we want to use LASSO regression in order to find genes of interest
# that could maybe later be used to compare with the findings of the autoencoder

# the process we use to select genes is sketchy though
# but we endup with decent resust with good ROC curve.



# load metadata and prepare dirrectory 
setwd("~/Library/CloudStorage/OneDrive-Personal/polimi/Thesis")

dir <- "./data/quant"
samples <- list.files(dir)
files <- file.path(dir, samples)



##### LEGACY
## Load data from previous files

#ds_BL_1  = readRDS("./workfiles/ds_BL_1.rds")
#ds_V02_1 = readRDS("./workfiles/ds_V02_1.rds")
#ds_V04_1 = readRDS("./workfiles/ds_V04_1.rds")
#ds_V06_1 = readRDS("./workfiles/ds_V06_1.rds")
#ds_V08_1 = readRDS("./workfiles/ds_V08_1.rds")


#ds_BL_2  = readRDS("./workfiles/ds_BL_2.rds")
#ds_V02_2 = readRDS("./workfiles/ds_V02_2.rds")
#ds_V04_2 = readRDS("./workfiles/ds_V04_2.rds")
#ds_V06_2 = readRDS("./workfiles/ds_V06_2.rds")
#ds_V08_2 = readRDS("./workfiles/ds_V08_2.rds")


ds = fread("./workfiles/scaled_data.csv")

p = dim(ds)[2]
p

setkey(ds,clinical_event,cohort) # prety quick operation
ds_BL_1 = ds[.("BL",1)][,6:p]
ds_BL_2 = ds[.("BL",2)][,6:p]




###############################################################################
############################## LASSO  regression ############################## 
###############################################################################


# let's keep track of the legnth of each population
p = dim(ds_BL_1)[2]
p
p = 95309
n1_1 = dim(ds_BL_1)[1]
#n2_1 = dim(ds_V02_1)[1]
#n3_1 = dim(ds_V04_1)[1]
#n4_1 = dim(ds_V06_1)[1]
#n5_1 = dim(ds_V08_1)[1]



# let's keep track of the length of each population

n1_2 = dim(ds_BL_2)[1]
#n2_2 = dim(ds_V02_2)[1]
#n3_2 = dim(ds_V04_2)[1]
#n4_2 = dim(ds_V06_2)[1]
#n5_2 = dim(ds_V08_2)[1]

cohort = as.matrix(c(rep(1, n1_1), rep(0, n1_2)), ncol = 1)
y = cohort

#data = readRDS("./workfiles/ds_BL_1_and_2.rds")

#data = rbind(ds_BL_1, ds_BL_2)
data <- rbindlist(list(ds_BL_1, ds_BL_2))



sums = data[, lapply(.SD, sum, na.rm=TRUE)]
# sums <- colSums(data) # way faster


class(sums)
plot(density(as.numeric(sums)))

# get the list of genes real quick
genes = read.delim(files[10], header = TRUE)[,1]
genes_selected = genes#[sums > 10000]
(length(genes_selected))

###############################################################################
############################### reduce the data ############################### 
###############################################################################

# LASSO
#?glmnet


#X = data#[, sums > 10000]
X = as.matrix(data)#[, sums > 10000]




###############################################################################
################################ fit the model ################################
###############################################################################


# alpha = 1 for LASSO
cv.lambda.lasso = cv.glmnet(X, y, family = "binomial", alpha = 1, nlambda = 50)
l.lasso.min <- cv.lambda.lasso$lambda.min
l.lasso.min
#0.02155367


plot(cv.lambda.lasso$glmnet.fit, "lambda", label=FALSE)
plot(cv.lambda.lasso)

lasso.model <- glmnet(x=X, y=y,
                      alpha  = 1, 
                      family = "binomial",
                      lambda = l.lasso.min)

betas = as.vector(lasso.model$beta)                         #Coefficients


# the genes of interest
genes_selected[betas != 0]

# main ones:

# WARNING : is the rank function doing what we want to do ?????
genes_selected[rank(genes_selected[betas != 0])] [1:5]


# check the summary
summary(lasso.model)
plot(lasso.model)    # why nothing is shown ?


assess.glmnet(lasso.model,           #in this case, we are evaluating the model
              newx = X,              #in the same data used to fit the model
              newy = y )             #so newx=X and newY=Y

plot(roc.glmnet(lasso.model, 
                newx = X, 
                newy = as.factor(y) ), 
     type="l")                       #produces the ROC plot











