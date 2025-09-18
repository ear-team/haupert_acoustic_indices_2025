#==================================================================================
#
# Find the correlation between the indices that correlate with species richness
#
# => select the indices that exhibit low colinearity in order to be used in the
# in the modelling process (linear regression using a vector of indices)
# Author: Sylvain Haupert
# Date: 2025-01-20
#==================================================================================

# LOAD LIBRARIES 
library(data.table)
library(corrplot)

# OPTIONS
CORRELATION_TYPE = 'spearman'

# IMPORT CSV (you may need to adapt the path depending on your working directory)
data = fread("./results/data_species.csv")

# Extract the acoustic index only
data_index = data[,c(2:61)]

# select the columns by their name
# data_index_selection = data_index [,c('ZCR', 'EVNtMean', 'MEANf', 'SKEWf', 'KURTf', 'NP', 'SNRf', 'Hf', 'H', 'EAS', 'ECU', 
#                                 'ECV', 'EPS', 'ACI', 'NDSI', 'rBA', 'BioEnergy', 'BIO', 'LFC', 'MFC', 'ACTspFract', 
#                                 'ACTspCount', 'ACTspMean', 'EVNspFract', 'EVNspMean', 'EVNspCount', 'TFSD', 
#                                 'H_Havrda', 'H_Renyi', 'H_pairedShannon', 'H_GiniSimpson', 'AGI', 'nROI', 'aROI')]

data_index_selection = data_index [,c('ACTtFraction', 'ACTtCount', 'EVNtFraction', 'EVNtMean', 'EVNtCount', 
                                'MEANf', 'SKEWf', 'KURTf', 'NP', 'SNRf', 'Hf', 'H', 'EAS', 'ECU', 'ECV', 
                                'EPS', 'EPS_KURT', 'EPS_SKEW', 'ACI', 'NDSI', 'rBA', 'BioEnergy', 'BIO', 
                                'ADI', 'AEI', 'LFC', 'MFC', 'ACTspFract', 'ACTspCount', 'ACTspMean', 'EVNspFract', 
                                'EVNspMean', 'EVNspCount', 'TFSD', 'H_Havrda', 'H_Renyi', 'H_pairedShannon', 
                                'H_GiniSimpson', 'AGI', 'nROI', 'aROI')]

# Resize the figure 
options(repr.plot.width = 20, repr.plot.height = 15, repr.plot.res = 300)

# I want a color palette with 10 colors (in order to have a step of 0.25) that I choose
# I set |R|>0.75 to dark blue or red colors for easy reading
col = c("#003881d2", "#97deff96", "#97deff96","#97deff96", "#ff826c86", "#ff826c86", "#ff826c86",  "#b82b00e8")

# Compute the correlation matrix
cor_matrix = cor(data_index_selection, method=CORRELATION_TYPE, use="pairwise.complete.obs")

# compute the statistics and p-value
pvalue_matrix = cor.mtest(data_index_selection, conf.level = 0.95)$p

# plot the correlation matrix with the p-value
corrplot_obj = corrplot(
        # corr=cor_matrix[manual_order,manual_order],
        corr=cor_matrix,
        p.mat = pvalue_matrix,
        order = 'hclust',
        method = 'circle',
        insig = 'label_sig',
        sig.level = c(0.001, 0.01, 0.05),
        col = col,
        pch.cex = 0.7,
        pch.col = '#1a1a1a8c',
        tl.col = '#000000',
        tl.srt = 45,
        tl.cex = 0.66,
        cl.ratio =0.1,
        cl.cex = 0.66
        )

# Save the figure in png format
png(filename = "./results/figure_6_onlybirds.png", width = 20, height = 15, units = "cm", res = 300)
dev.off()
