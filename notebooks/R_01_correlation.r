# ==================================================================================
#
# Find the correlation between the indices that correlate with species richness
#
# => select the indices that exhibit low colinearity in order to be used in the
# in the modelling process (linear regression using a vector of indices)
# Author: Sylvain Haupert
# Date: 2025-01-20
# ==================================================================================

# LOAD LIBRARIES
if (!require("data.table")) install.packages("data.table")
if (!require("corrplot")) install.packages("corrplot")
library(data.table)
library(corrplot)

# OPTIONS
CORRELATION_TYPE <- "spearman"

# IMPORT CSV (you may need to adapt the path depending on your working directory)
data <- fread("./results/train_dataset_for_statistical_modeling_in_R.csv")

# List of index with |R|>0.3
LIST_INDICES = c('nROI', 'aROI', 'NP', 'EAS', 'EPS', 'ACI', 'NDSI', 'rBA', 
                'BioEnergy', 'BIO', 'LFC', 'MFC', 'ACTspFract', 
                'ACTspCount', 'ACTspMean', 'EVNspFract', 'EVNspMean', 
                'EVNspCount', 'TFSD', 'AGI' )

# Extract the acoustic index only
data_index <- data[, c(2:61)]

# select the columns by their name
data_index_selection <- data_index[, LIST_INDICES]

# Resize the figure
options(repr.plot.width = 20, repr.plot.height = 15, repr.plot.res = 300)

# I want a color palette with 10 colors (in order to have a step of 0.25) that I choose
# I set |R|>0.75 to dark blue or red colors for easy reading
col <- c("#003881d2", "#97deff96", "#ff826c86", "#b82b00e8")

# Compute the correlation matrix
cor_matrix <- cor(data_index_selection, method = CORRELATION_TYPE, use = "pairwise.complete.obs")

# compute the statistics and p-value
pvalue_matrix <- cor.mtest(data_index_selection, conf.level = 0.95)$p

# keep only positive correlations and clip to [0, 1]
cor_matrix_pos <- cor_matrix
cor_matrix_pos[cor_matrix_pos < 0] <- 0 # set negatives to 0

# Force scale range by temporarily setting one diagonal element to 0
# (will be overwritten by clustering but ensures color scale starts at 0)
# cor_matrix_pos[1, 1] <- 0

# Find the lowest value and set it to 0
cor_matrix_pos[which(cor_matrix_pos == min(cor_matrix_pos))] <- 0

# Save the figure in png format
png(filename = "./results/figure_S11.png", width = 20, height = 15, units = "cm", res = 300)

# plot the correlation matrix with the p-value
corrplot_obj <- corrplot(
        # corr=cor_matrix[manual_order,manual_order],
        corr = cor_matrix_pos,
        p.mat = pvalue_matrix,
        order = "hclust",
        method = "circle",
        insig = "label_sig",
        sig.level = c(0.001, 0.01, 0.05),
        col = col,
        pch.cex = 0.7,
        pch.col = "#1a1a1a8c",
        tl.col = "#000000",
        tl.srt = 45,
        tl.cex = 0.66,
        cl.ratio = 0.2,
        cl.cex = 0.66,
        is.corr = FALSE # allow custom range
)

dev.off()

# VIF (Variation Inflation Factor) calculation to check for multicollinearity
# ===============================================================================
if (!require("car")) install.packages("car")
library(car)

# Create a dataset that includes species_richness for VIF analysis
data_for_vif <- data[, c("species_richness", LIST_INDICES), with = FALSE]
# compute a simple linear model
vif_model <- lm(species_richness ~ ., data = data_for_vif)
# calculate VIF values
vif_values <- vif(vif_model)
print(vif_values)

# remove in the list the indices that have a VIF > 10 (common threshold for multicollinearity)
LIST_INDICES_REDUCED <- LIST_INDICES[vif_values <= 10]
# compute the VIF values again with the reduced list of indices
data_for_vif_reduced <- data[, c("species_richness", LIST_INDICES_REDUCED), with = FALSE]
vif_model_reduced <- lm(species_richness ~ ., data = data_for_vif_reduced)
vif_values_reduced <- vif(vif_model_reduced)
print(vif_values_reduced)
