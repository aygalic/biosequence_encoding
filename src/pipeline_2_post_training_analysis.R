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


setwd("~/Thesis/genome_analysis_parkinson/src")


# let's work with a somewhat decent dataframe first
table = fread("../workfiles/processed_data.csv", header = T)





names = table$name
names


encoded_experession = table[,1:256] 



#################################
################################# PCA
#################################



# autoencoder are a sort of non linear PCA
# let's see if PCA does anything 

pc.data = princomp(encoded_experession, scores=T)

plot(pc.data) # this is a very basic classifier and the PCA is able to enhance it a lot


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
phases <- sapply(names, function(names) c(strsplit(names, "-", fixed = T)[[1]][2]), USE.NAMES=FALSE)
time_points <- sapply(names, function(names) c(strsplit(names, ".", fixed = T)[[1]][3]), USE.NAMES=FALSE)


plot(projected_data, col = factor(time_points), pch = 16)
plot(projected_data, col = factor(cohorts), pch = 16)

legend_ = c("Parkinson's Disease", "Healthy Control", "SWEDD", "Prodromal")
legend("topleft",
       legend = legend_,
       pch = 19,
       col = factor(levels(factor(cohorts))))





#################################
################################# animated t-SNE
#################################


data_matrix <- as.matrix(encoded_experession)
perplexities = c(2, 3, 4, 5, 10, 25, 50, 75, 100)
iter = c(100, 200, 300, 400, 500, 700)
levels = as.factor(cohorts)
levels = as.factor(time_points)





make_animated_plot = function(param){
  
  # initialize the thing
  tsne_out <- Rtsne(data_matrix,perplexity = param, pca = FALSE, max_iter = 1)
  tsne_plots <- data.frame(x = tsne_out$Y[,1],
                           y = tsne_out$Y[,2],
                           max_iter = 1,
                           group = levels)
  for(i in iter){
    tsne_out <- Rtsne(data_matrix,perplexity = param, pca = FALSE, max_iter = i)
    tsne_plot <- data.frame(x = tsne_out$Y[,1],
                            y = tsne_out$Y[,2],
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



plots <- map(perplexities, make_animated_plot, .progress = TRUE) # use map instead for efficiency ?
subplot(plots, nrows = 3)






