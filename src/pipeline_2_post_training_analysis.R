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
library(data.table)

setwd("~/Thesis/genome_analysis_parkinson/src")

table = fread("../workfiles/processed_data.csv", header = T)
table = fread("../workfiles/processed_data_untransformed.csv", header = T)

table = fread("../workfiles/processed_data_vae.csv", header = T)

# these are the file names for each encoded observation
names = table$name


#####
#####
#####



p = dim(table)[2] - 1
n = dim(table)[1] 
# this is the corresponding encoding
encoded_experession = table[,1:p] # PAY ATTENTION TO THIS, maybe it needs to be [,1:p]


#################################
################################# PCA
#################################
# we start with PCA as it is the most vanilla approach to visualization


# autoencoder are a sort of non linear PCA
# let's see if PCA does anything 

pc.data = princomp(encoded_experession, scores=T)

plot(pc.data) 


# proportion of variance explained by each PC
plot(pc.data$sd^2/sum(pc.data$sd^2))
# cumulative proportion of explained variance
plot(cumsum(pc.data$sd^2)/sum(pc.data$sd^2))


# let's visualize the data over 
projected_data = as.matrix(encoded_experession) %*% as.matrix(pc.data$loadings[,1:2])

plot(projected_data) # it would be nice to know which patient correspond to what

meta <- read_excel("../../METADATA_200123.xlsx", sheet = "Foglio1")
patient_ids <- sapply(names, function(names) c(strsplit(names, ".", fixed = T)[[1]][2]), USE.NAMES=FALSE)
cohorts = meta$Cohort[match(patient_ids, meta$`Patient Number`)]
DS = meta$`Disease Status`[match(patient_ids, meta$`Patient Number`)]
phases <- sapply(names, function(names) c(strsplit(names, "-", fixed = T)[[1]][2]), USE.NAMES=FALSE)
time_points <- sapply(names, function(names) c(strsplit(names, ".", fixed = T)[[1]][3]), USE.NAMES=FALSE)

dim(encoded_experession)
length(cohorts)

xlim_ = c(min(projected_data[,1]), max(projected_data[,1]))
ylim_ = c(min(projected_data[,2]), max(projected_data[,2]))


plot(projected_data, col = factor(time_points), pch = 16)
plot(projected_data, col = factor(cohorts), pch = 16)

plot(projected_data, col = factor(cohorts), pch = 16, xlim = xlim_, ylim = ylim_)



legend_ = c("Parkinson's Disease", "Healthy Control", "SWEDD", "Prodromal")
legend("topleft",
       legend = legend_,
       pch = 19,
       col = factor(levels(factor(cohorts))))


plot(projected_data, col = factor(cohorts), pch = 16)



# here we have some decent separation between PD and Prodromal
plot(projected_data, col = factor(cohorts), pch = 16)

plot( projected_data[cohorts %in% c(1,4),],
      col = factor(cohorts[cohorts %in% c(1,4)]), 
      pch = 16, xlim = xlim_, ylim = ylim_)

plot( projected_data[cohorts %in% c(1),],
      col = factor(cohorts[cohorts %in% c(1)]), 
      pch = 16, xlim = xlim_, ylim = ylim_)

plot( projected_data[cohorts %in% c(2),],
      col = factor(cohorts[cohorts %in% c(2)]), 
      pch = 16, xlim = xlim_, ylim = ylim_)

plot( projected_data[cohorts %in% c(3),],
      col = factor(cohorts[cohorts %in% c(3)]), 
      pch = 16, xlim = xlim_, ylim = ylim_)

plot( projected_data[cohorts %in% c(4),],
      col = factor(cohorts[cohorts %in% c(4)]), 
      pch = 16, xlim = xlim_, ylim = ylim_)



#################################
################################# animated t-SNE
#################################


data_matrix <- as.matrix(encoded_experession)
perplexities = c(2, 3, 4, 5, 10, 25, 50, 75, 100)
iter = c(100, 200, 300, 400, 500, 700, 2000)

levels = as.factor(time_points)
levels = as.factor(cohorts)





make_animated_plot = function(param){
  
  # initialize the thing
  tsne_out <- Rtsne(data_matrix,perplexity = param, pca = FALSE, max_iter = 1)
  tsne_plots <- data.frame(x = rescale(tsne_out$Y[,1]), # rescale for viz
                           y = rescale(tsne_out$Y[,2]),
                           max_iter = 1,
                           group = levels)
  for(i in iter){
    tsne_out <- Rtsne(data_matrix,perplexity = param, pca = FALSE, max_iter = i)
    tsne_plot <- data.frame(x = rescale(tsne_out$Y[,1]),
                            y = rescale(tsne_out$Y[,2]),
                            max_iter = i,
                            group = levels)
    tsne_plots <- rbind(tsne_plots, tsne_plot)
  }
  plot = ggplot(tsne_plots, aes(x=x, y=y, col = group, frame = max_iter)) + 
    geom_point() + 
    theme_void() +
    scale_color_viridis(discrete = TRUE, option = "A")  # A for mamgma colors
  plot <- ggplotly(plot)
  return(plot)
}



plots <- map(perplexities, make_animated_plot, .progress = TRUE) # use map for efficiency 
subplot(plots, nrows = 3)





# if you want to investigate a given plot : 

group = as.factor(time_points)
group = as.factor(cohorts)


tsne_out <- Rtsne(data_matrix,perplexity = 3, pca = FALSE, max_iter = 100)
plt_data <- data.frame(x = rescale(tsne_out$Y[,1]), # rescale for viz
                         y = rescale(tsne_out$Y[,2]))

plot = ggplot(plt_data, aes(x=x, y=y, col = group)) + 
 geom_point(size = 1) + 
 theme_void() +
 scale_color_viridis(discrete = TRUE, option = "A")  # A for mamgma colors
plot <- ggplotly(plot)
plot                         


orca(plot, "../img/t-SNE_simple_ae_log1p.png") 

                         
