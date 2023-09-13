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
library("rjson")


setwd("~/Thesis/genome_analysis_parkinson/src")


metadata = fromJSON(file = "/Users/aygalic/Thesis/data/cancer/metadata.cart.2023-09-11.json")
clinical = read.delim(file = "/Users/aygalic/Thesis/data/cancer/clinical.cart.2023-09-11/clinical.tsv")


# doing some testing to understand the data structure
metadata[[1]]$file_name
metadata[[2]]$file_name

metadata[[1]]$associated_entities[[1]]$case_id

case_id = metadata[[1]]$associated_entities[[1]]$case_id
case_id

clinical[clinical$case_id == case_id,]$case_id
clinical[clinical$case_id == case_id,]$tissue_or_organ_of_origin




# loading encoded files
table = fread("../workfiles/FC_ae_cancer.csv", header = T)


# these are the file names for each encoded observation
names = table$name
names[1:5]

strsplit(names[1], "/")[[1]][8]

# getting just the filename out of the entire path
name_process <- function(name){
  return(strsplit(name, "/")[[1]][8])
}


filenames = map_vec(names, name_process, .progress = T)
filenames[1:5]


# now we retrieve the corresponding clinical info for each file
get_clinical_info <- function(given_filename){
  
  # Initialize a variable to store the index of the matching entity
  matching_index <- NULL
  
  # Loop through the metadata list
  for (i in seq_along(metadata)) {
    if (metadata[[i]]$file_name == given_filename) {
      matching_index <- i
      break  # Stop the loop when a match is found
    }
  }
  case_id = metadata[[matching_index]]$associated_entities[[1]]$case_id
  return(clinical[clinical$case_id == case_id,]$tissue_or_organ_of_origin)
  
}

clinical_info <- map_vec(filenames, get_clinical_info, .progress = T)
clinical_info[1:5]


#####
#####
#####



p = dim(table)[2] - 1
n = dim(table)[1] 
# this is the corresponding encoding
encoded_experession = table[,1:p]



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


dim(encoded_experession)
length(clinical_info)




#### let's plot the PCA for the most abundant tissues
plot_df <- data.frame(projected_data)


n_tissues = 5



# Step 1: Count the occurrences of each factor
factor_counts <- table(clinical_info)

# Step 2: Sort the factor counts in descending order
sorted_counts <- sort(factor_counts, decreasing = TRUE)

# Step 3: Extract the names of the top n factors
top_factors <- names(sorted_counts[1:n_tissues])

# Step 4: Create a new factor variable that groups other factors as "other"
plot_df$tissue_of_origin <- ifelse(clinical_info %in% top_factors, clinical_info, "_other")


# A basic scatterplot with color depending on tissue
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

pal <- gg_color_hue(n_tissues + 1)
pal[1] <- "#000000"

  
ggplot(plot_df, aes(x = Comp.1,y = Comp.2, color = tissue_of_origin)) + 
  scale_color_manual(values=pal)+
  geom_point(size=1) + 
  theme_void()









#################################
################################# animated t-SNE
#################################


data_matrix <- as.matrix(encoded_experession)
perplexities = c(2, 3, 4, 5, 10, 25, 50, 75, 100)
iter = c(100, 200, 300, 400, 500, 700, 2000)

levels = as.factor(clinical_info)





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
    theme_void() #+
    #scale_color_viridis(discrete = TRUE, option = "A")  # A for mamgma colors
  plot <- ggplotly(plot)
  return(plot)
}



plots <- map(perplexities, make_animated_plot, .progress = TRUE) # use map for efficiency 
subplot(plots, nrows = 3)





# if you want to investigate a given plot : 
group = as.factor(clinical_info)
perplexity = 50
iter_max = 2000

tsne_out <- Rtsne(data_matrix,perplexity = perplexity, pca = FALSE, max_iter = iter_max)
plt_data <- data.frame(x = tsne_out$Y[,1],
                       y = tsne_out$Y[,2])

plt_data$tissue_of_origin <- ifelse(clinical_info %in% top_factors, clinical_info, "_other")

plot = ggplot(plt_data, aes(x=x, y=y, color = tissue_of_origin)) + 
  geom_point(size = 1) + 
  scale_color_manual(values=pal)+
  theme_void()
plot 



                         
