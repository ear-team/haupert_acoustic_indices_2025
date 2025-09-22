# =============================================================================
# Statistical Modeling with Generalized Linear Mixed Models (GLMM) in R
# Translation from Python code with proper random effects implementation
# Updated: January 2025
# =============================================================================

# Clear workspace
rm(list = ls())

# set wd to the script location
setwd("/media/haupert/data/mes_projets/07_nroi/nROI_publication.git/haupert_acoustic_indices_2025/notebooks/R_02_statistical_model.r")

# Load required libraries
# =============================================================================
if (!require("lme4")) install.packages("lme4")
if (!require("glmmTMB")) install.packages("glmmTMB")
if (!require("DHARMa")) install.packages("DHARMa")
if (!require("sjPlot")) install.packages("sjPlot")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("broom.mixed")) install.packages("broom.mixed")
if (!require("performance")) install.packages("performance")
if (!require("MuMIn")) install.packages("MuMIn")
if (!require("emmeans")) install.packages("emmeans")
if (!require("car")) install.packages("car")

library(lme4) # Mixed-effects models
library(glmmTMB) # Generalized linear mixed models with various distributions
library(DHARMa) # Residual diagnostics for GLMMs
library(sjPlot) # Tables and plots for mixed models
library(ggplot2) # Plotting
library(dplyr) # Data manipulation
library(broom.mixed) # Tidy mixed model outputs
library(performance) # Model performance metrics
library(MuMIn) # Model selection and averaging
library(emmeans) # Estimated marginal means
library(car) # Anova for mixed models

# Set options
# =============================================================================
options(contrasts = c("contr.sum", "contr.poly")) # Type III sum of squares
options(width = 120)

# Data loading and preparation
# =============================================================================
cat("Loading and preparing data...\n")

# Load the data
data_path <- "./results/data_statistical_model.csv"
if (!file.exists(data_path)) {
    stop("Data file not found. Please check the path: ", data_path)
}

data <- read.csv(data_path, stringsAsFactors = FALSE)

# Display data structure
cat("Data structure:\n")
str(data)
cat("\nFirst few rows:\n")
head(data)

# Data preprocessing
# =============================================================================
cat("\nData preprocessing...\n")

# Convert categorical variables to factors
data$device_id <- as.factor(data$device_id)
data$site <- as.factor(data$site)
data$habitat <- as.factor(data$habitat)
data$dataset <- as.factor(data$dataset)

# Check for missing values
cat("Missing values per column:\n")
print(colSums(is.na(data)))

# Remove rows with missing values in key variables
data_clean <- data[complete.cases(data[c("species_richness", "nROI", "device_id", "habitat", "dataset")]), ]

cat("Data dimensions after cleaning:", nrow(data_clean), "rows,", ncol(data_clean), "columns\n")

# Aggregate data by site (as in Python code)
# =============================================================================
cat("\nAggregating data by site...\n")

# Average species richness and nROI per site keeping site, habitat, device_id, and dataset
data_agg <- data_clean %>%
    group_by(site, habitat, device_id, dataset) %>%
    summarise(
        species_richness = mean(species_richness, na.rm = TRUE),
        nROI = mean(nROI, na.rm = TRUE),
        n_observations = n(),
        .groups = "drop"
    )

cat("Aggregated data dimensions:", nrow(data_agg), "rows,", ncol(data_agg), "columns\n")

# Summary statistics
cat("\nSummary statistics:\n")
print(summary(data_agg))

# Display data distribution by groups
cat("\nData distribution by groups:\n")
cat("Number of observations per dataset:\n")
print(table(data_agg$dataset))
cat("\nNumber of observations per habitat:\n")
print(table(data_agg$habitat))
cat("\nNumber of observations per device:\n")
print(table(data_agg$device_id))

# Exploratory data analysis
# =============================================================================
cat("\nExploratory Data Analysis...\n")

# Distribution of species richness
p1 <- ggplot(data_agg, aes(x = species_richness)) +
    geom_histogram(bins = 10, fill = "skyblue", alpha = 0.7) +
    labs(title = "Distribution of Species Richness", x = "Species Richness", y = "Frequency") +
    theme_minimal()

print(p1)

# Distribution of nROI
p1_nroi <- ggplot(data_agg, aes(x = nROI)) +
    geom_histogram(bins = 10, fill = "lightgreen", alpha = 0.7) +
    labs(title = "Distribution of nROI", x = "nROI", y = "Frequency") +
    theme_minimal()
    
print(p1_nroi)

cat("\nBoth species richness and nROI distributions seem multimodal 
    \n A Gaussian mixture model could be appropriate.\n")

# NOTE: There is no "Gaussian mixture" family in glm() or glmmTMB().
# Mixture models are not supported as a 'family' in standard regression functions.
# Closest approach: Fit a Gaussian mixture model separately (e.g., with mclust or mixtools),
# or use flexible regression (e.g., GAMs) if you want to model multimodality in the response.

# Example: Fit a Gaussian mixture model to species_richness (for visualization only)
if (!require("mclust")) install.packages("mclust")
library(mclust)
gmm_fit <- Mclust(data_agg$species_richness, G = 2:4)
cat("Best number of mixture components (BIC):", gmm_fit$G, "\n")
plot(gmm_fit, what = "density")

# Predict cluster membership
data_agg$gmm_cluster <- predict(gmm_fit)$classification
cat("GMM cluster membership added to data_agg\n")
# Visualize clusters
p_gmm <- ggplot(data_agg, aes(x = nROI, y = species_richness, color = as.factor(gmm_cluster))) +
    geom_point(alpha = 0.7) +
    labs(title = "GMM Clusters of Species Richness vs nROI", x = "nROI", y = "Species Richness", color = "GMM Cluster") +
    theme_minimal()
print(p_gmm)

# For regression: You can fit a standard Gaussian GLM as below,
# but it will not capture multimodality in the response distribution.
p2 <- ggplot(data_agg, aes(x = nROI, y = species_richness)) +
    geom_point(aes(color = habitat), alpha = 0.7) +
    geom_smooth(method = "glm", method.args = list(family = gaussian(link = "identity")), se = TRUE) +
    labs(title = "Species Richness vs nROI (Gaussian GLM)", x = "nROI", y = "Species Richness") +
    theme_minimal() +
    facet_wrap(~dataset, scales = "free")
print(p2)

# Relationship between nROI and species richness but by habitat
# Faceted by habitat
p2_habitat <- ggplot(data_agg, aes(x = nROI, y = species_richness)) +
    geom_point(aes(color = habitat), alpha = 0.7) +
    geom_smooth(method = "glm", method.args = list(family = gaussian(link = "identity")), se = TRUE) +
    labs(title = "Species Richness vs nROI by Habitat", x = "nROI", y = "Species Richness") +
    theme_minimal() +
    facet_wrap(~habitat, scales = "free")
print(p2_habitat)   

# Box plots by groups
p3 <- ggplot(data_agg, aes(x = habitat, y = species_richness)) +
    geom_boxplot(aes(fill = habitat), alpha = 0.7) +
    labs(title = "Species Richness by Habitat", x = "Habitat", y = "Species Richness") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p3)

# Model fitting
# =============================================================================
cat("\n" %+% "=" %+% rep("=", 60) %+% "\n")
cat("STATISTICAL MODELING\n")
cat("=" %+% rep("=", 60) %+% "\n")

# Check distribution of species richness for model selection
cat("Species richness distribution for model selection:\n")
print(summary(data_agg$species_richness))
cat("Variance:", var(data_agg$species_richness), "\n")
cat("Mean:", mean(data_agg$species_richness), "\n")
cat("Variance/Mean ratio:", var(data_agg$species_richness) / mean(data_agg$species_richness), "\n")

# Check if variables are appropriate for different model types
cat("\nData type check AFTER aggregation:\n")
cat("Species richness - integers?", all(data_agg$species_richness == round(data_agg$species_richness), na.rm = TRUE), "\n")
cat("Species richness range:", range(data_agg$species_richness), "\n")
cat("Any zero values?", any(data_agg$species_richness == 0), "\n")
cat("nROI - integers?", all(data_agg$nROI == round(data_agg$nROI), na.rm = TRUE), "\n")
cat("nROI range:", range(data_agg$nROI), "\n")
cat("Any zero values?", any(data_agg$nROI == 0), "\n")

# Based on the above checks:
# - Species richness and nROI are continuous and positive (with zeros) but multimodal
# - GLM with Gaussian family is a starting point, but may not capture multimodality.
#   Moreover, GLM was not able to converge well with random effects...
# - We preferred to consider advanced models (e.g., mixture models) such as GAMMs

# GLM assumes: Y ~ Normal(μ, σ²) [unimodal]
# But our data: Y ~ π₁N(μ₁,σ₁²) + π₂N(μ₂,σ₂²) + ... [multimodal]

# Model 1: GAMM (Generalized Additive Mixed Model)
# Note: GAMMs can handle non-linear relationships and multimodal distributions better.
if (!require("mgcv")) install.packages("mgcv")
library(mgcv)
if (!require("mgcViz")) install.packages("mgcViz")
library(mgcViz)
cat("\nFitting GAMM...\n")

# Fit the Generalized Additive Mixed Model
# When variables are crossed (device_id and habitat could belong to different datasets), 
# we must include each of them as its own independent random effect
# to account for the overall variability they each contribute. 
# The correct model is the one that treats each variable as an independent random effect.
gamm_model1 <- gam(
    species_richness ~ 
    s(nROI) +                       # Smooth term for nROI
    s(habitat, bs = "re") +         # Random effect for habitat
    s(device_id, bs = "re")+        # Random effect for device_id
    s(dataset, bs = "re"),          # Random effect for dataset
    data = data_agg,                # Data frame
    # family = negative.binomial(link = "log", theta = 1), # Negative Binomial for overdispersed count data
    method = "REML"                 # Restricted Maximum Likelihood for smoothing parameter estimation
)
summary(gamm_model1)
plot(gamm_model1, pages = 1)
# Check residuals
gamm_res1 <- simulateResiduals(gamm_model1)
plot(gamm_res1)
cat("GAMM model 1 fitted and diagnostics checked.\n")

# Minimal model with only nROI to cpmpare
gamm_model_min <- gam(
    species_richness ~ 
    s(nROI),                        # Smooth term for nROI
    data = data_agg,                # Data frame
    method = "REML"                 # Restricted Maximum Likelihood for smoothing parameter estimation
)
summary(gamm_model_min)
plot(gamm_model_min, pages = 1)
# Check residuals
gamm_res_min <- simulateResiduals(gamm_model_min)
plot(gamm_res_min)
cat("Minimal GAMM model fitted and diagnostics checked.\n") 

# Model comparison
cat("\nComparing GAMM models...\n")

cat("\n Deviance explained:\n")
cat("Full model 1:", summary(gamm_model1)$dev.expl, "\n")
cat("Minimal model:", summary(gamm_model_min)$dev.expl, "\n")

cat("\n R² adjusted:\n")
cat("Full model 1:", summary(gamm_model1)$r.sq, "\n")
cat("Minimal model:", summary(gamm_model_min)$r.sq, "\n")

AIC_gamm <- AIC(gamm_model1, gamm_model2, gamm_model_min)
print(AIC_gamm)
cat("Lower AIC indicates a better model fit.\n")
cat("GAMM modeling complete.\n")
# =============================================================================

# Spatial autocorrelation analysis (if spatial coordinates are available)
# =============================================================================
# If your dataset includes spatial coordinates (e.g., latitude and longitude),
# you can assess spatial autocorrelation in the residuals of your model.
# Uncomment and modify the following code as needed.
# if (!require("spdep")) install.packages("spdep")
# library(spdep)