library(mvtnorm)

library(rgl)
library(MASS)

library("readxl")

setwd("~/Library/CloudStorage/OneDrive-Personal/polimi/Thesis")

df = read.csv("./workfiles/compressed_data.csv")
#raw_df = read.csv("./workfiles/raw_data.csv", header = T)


#dim(raw_df)
names = df$name
df = df[,1:63]
#df = raw_df[, 7:95315]
df



#################################
################################# PCA
#################################



# autoencoder are a sort of non linear PCA
# let's see if PCA does anything 

pc.data = princomp(df, scores=T)
#pc.data = prcomp(df, scores=T)
plot(pc.data) # this is a very basic classifier and the PCA is able to enhance it a lot


# proportion of variance explained by each PC
plot(pc.data$sd^2/sum(pc.data$sd^2))
# cumulative proportion of explained variance
plot(cumsum(pc.data$sd^2)/sum(pc.data$sd^2))


# let's visualize the data over 
projected_data = as.matrix(df) %*% as.matrix(pc.data$loadings[,1:2])

plot(projected_data) # it would be nice to know which patient correspond to what

meta <- read_excel("METADATA_200123.xlsx", sheet = "Foglio1")
patient_ids <- sapply(names, function(names) c(strsplit(names, ".", fixed = T)[[1]][2]), USE.NAMES=FALSE)
patient_ids


cohorts = meta$Cohort[match(patient_ids, meta$`Patient Number`)]
cohorts


plot(projected_data, col = factor(cohorts), pch = 16)

legend_ = c("Parkinson's Disease", "Healthy Control", "SWEDD", "Prodromal")
legend("topleft",
       legend = legend_,
       pch = 19,
       col = factor(levels(factor(cohorts))))



#################################
################################# Unsupervised Clustering
#################################


# Let's see if clusters currectly emerges from the comrpessed data
hc.complete <- hclust(dist(df), method="complete")
hc.average <- hclust(dist(df), method="average")
hc.single <- hclust(dist(df), method="single")


plot(hc.complete) # best looking so far
plot(hc.average)
plot(hc.single)

rect.hclust(hc.complete, k=2)


# Fix k=4 clusters:
cutree(hc.complete, 4)

# Compare with k-means results
table(cutree(hc.complete, 4), cohorts) # comparison with cohorts
table(cutree(hc.average, 4), cohorts) # comparison with cohorts
table(cutree(hc.single, 4), cohorts) # comparison with cohorts

# it's very bad, but it is what we'd expect though 


# just for the sake of knowing how it looks like
plot(projected_data, col = factor(cutree(hc.complete, 4)), pch = 16)
# pretty much what you expect from this kind of techniqes


##### Let's see with kmeans
fit <- kmeans((df),4,nstart = 20)
table(fit$cluster, cohorts) # comparison with cohorts


plot(projected_data, col = factor(fit$cluster), pch = 16)
# same issue
# it's just not better



par(mfrow=c(2,2))
plot(projected_data, col = factor(cutree(hc.complete, 4)), pch = 16, main ="complete euclidiean")
plot(projected_data, col = factor(cutree(hc.average, 4)), pch = 16, main ="average euclidiean")
plot(projected_data, col = factor(cutree(hc.single, 4)), pch = 16, main ="single euclidiean")
plot(projected_data, col = factor(fit$cluster), pch = 16, main ="kmeans")
#plot(projected_data, col = factor(cohorts), pch = 16, main ="ACTUAL")




