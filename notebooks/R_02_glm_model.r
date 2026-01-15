# =============================================================================
# Statistical Modeling with Generalized Linear Mixed Models (GLMM) in R
# Updated: January 2026
# =============================================================================

# Clear workspace
rm(list = ls())

# set wd to the script location
setwd("./")

# options
# =============================================================================
SAVE_FIGURES <- FALSE # Set to TRUE to save figures, FALSE to only display them

# Load required libraries
# =============================================================================
if (!require("glmmTMB")) install.packages("glmmTMB")
if (!require("DHARMa")) install.packages("DHARMa")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("performance")) install.packages("performance")
if (!require("DescTools")) install.packages("DescTools")
if (!require("gridExtra")) install.packages("gridExtra")

library(glmmTMB) # Generalized linear mixed models with various distributions
library(DHARMa) # Residual diagnostics for GLMMs
library(ggplot2) # Plotting
library(dplyr) # Data manipulation
library(performance) # Model performance metrics
library(DescTools) # For CCC calculation
library(gridExtra) # For arranging multiple ggplots

# Set options
# =============================================================================
options(contrasts = c("contr.sum", "contr.poly")) # Type III sum of squares
options(width = 120)

# Data loading and preparation
# =============================================================================

# Load the data
data_path <- "./results/train_dataset_for_statistical_modeling_in_R.csv"
if (!file.exists(data_path)) {
    stop("Data file not found. Please check the path: ", data_path)
}
data <- read.csv(data_path, stringsAsFactors = FALSE)

# Display data structure
str(data)
head(data)

# Data preprocessing
# =============================================================================

# Convert categorical variables to factors
data$device_id <- as.factor(data$device_id)
data$site <- as.factor(data$site)
data$habitat <- as.factor(data$habitat)
data$dataset <- as.factor(data$dataset)

# Remove rows with missing values in key variables
data_clean <- data[complete.cases(data[c("species_richness", "nROI", "device_id", "habitat", "dataset")]), ]

cat("Data dimensions after cleaning:", nrow(data_clean), "rows,", ncol(data_clean), "columns\n")

# Aggregate data by site (as in Python code)
# =============================================================================

# Average species richness and nROI per site keeping site, habitat, device_id, and dataset
data_agg <- data_clean %>%
    group_by(site, habitat, device_id, dataset, LAT, LON) %>%
    summarise(
        species_richness = mean(species_richness, na.rm = TRUE),
        nROI = mean(nROI, na.rm = TRUE),
        n_observations = n(),
        .groups = "drop"
    )

cat("Aggregated data dimensions:", nrow(data_agg), "rows,", ncol(data_agg), "columns\n")

# Summary statistics
print(summary(data_agg))

# Display data distribution by groups
print(table(data_agg$dataset))
print(table(data_agg$habitat))
print(table(data_agg$device_id))

# Exploratory data analysis
# =============================================================================

# Distribution of species richness
p1 <- ggplot(data_agg, aes(x = species_richness)) +
    geom_histogram(bins = 20, fill = "skyblue", alpha = 0.7) +
    labs(title = "Distribution of Species Richness", x = "Average Species Richness by site", y = "Frequency") +
    theme_minimal()
print(p1)

# Distribution of nROI
p1_nroi <- ggplot(data_agg, aes(x = nROI)) +
    geom_histogram(bins = 20, fill = "lightgreen", alpha = 0.7) +
    labs(title = "Distribution of nROI", x = "Average nROI by site", y = "Frequency") +
    theme_minimal()
print(p1_nroi)

# create a new plot with p1 and p1_nROI.
p1_combined <- grid.arrange(p1_nroi, p1, ncol = 2)
print(p1_combined)

# save the plot
if (SAVE_FIGURES == TRUE) {
    ggsave("./results/figure_S4.png", plot = p1_combined, dpi = 300)
}

# Relationship between nROI and species richness but by habitat
# Faceted by habitat
p2_habitat <- ggplot(data_agg, aes(x = nROI, y = species_richness)) +
    geom_point(aes(color = habitat), alpha = 0.7) +
    geom_smooth(method = "glm", method.args = list(family = gaussian(link = "identity")), se = TRUE) +
    labs(title = "Species Richness vs nROI by Habitat", x = "nROI", y = "Species Richness") +
    theme_minimal() +
    facet_wrap(~habitat, scales = "free", ncol = 3) +
    coord_cartesian(xlim = c(0, 260), ylim = c(0, 8)) +
    theme(legend.position = c(0.82, 0.15))
print(p2_habitat)

# save the plot
if (SAVE_FIGURES == TRUE) {
    ggsave("./results/figure_S5.png", plot = p2_habitat, dpi = 300)
}

# For regression: You can fit a standard Gaussian GLM as below,
# but it will not capture multimodality in the response distribution.
# remove the legend for clarity
p2 <- ggplot(data_agg, aes(x = nROI, y = species_richness)) +
    geom_point(aes(color = habitat), alpha = 0.7) +
    geom_smooth(method = "glm", method.args = list(family = gaussian(link = "identity")), se = TRUE) +
    labs(title = "Species Richness vs nROI (Gaussian GLM)", x = "nROI", y = "Species Richness") +
    theme_minimal() +
    facet_wrap(~dataset, scales = "free") +
    coord_cartesian(xlim = c(0, 260), ylim = c(0, 8)) +
    theme(legend.position = "none")
print(p2)

# save the plot
if (SAVE_FIGURES == TRUE) {
    ggsave("./results/figure_S6.png", plot = p2, dpi = 300)
}

# Box plots by groups
p3 <- ggplot(data_agg, aes(x = habitat, y = species_richness)) +
    geom_boxplot(aes(fill = habitat), alpha = 0.7) +
    labs(title = "Species Richness by Habitat", x = "Habitat", y = "Species Richness") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
    scale_x_discrete(labels = function(x) gsub("-", "\n", x))

print(p3)

# Model fitting
# =============================================================================

# Keep habitats with at least 3 observations (n_observations)
data_agg <- data_agg %>%
    group_by(habitat) %>%
    filter(n() >= 3) %>%
    ungroup()

# remove empty habitats
data_agg <- droplevels(data_agg)

# Check distribution of species richness for model selection
print(summary(data_agg$species_richness))
cat("Variance:", var(data_agg$species_richness), "\n")
cat("Mean:", mean(data_agg$species_richness), "\n")
cat("Variance/Mean ratio:", var(data_agg$species_richness) / mean(data_agg$species_richness), "\n")

# Check if variables are appropriate for different model types
cat("Species richness - integers?", all(data_agg$species_richness == round(data_agg$species_richness), na.rm = TRUE), "\n")
cat("Species richness range:", range(data_agg$species_richness), "\n")
cat("Any zero values?", any(data_agg$species_richness == 0), "\n")
cat("nROI - integers?", all(data_agg$nROI == round(data_agg$nROI), na.rm = TRUE), "\n")
cat("nROI range:", range(data_agg$nROI), "\n")
cat("Any zero values?", any(data_agg$nROI == 0), "\n")
cat("Variance/Mean ratio:", var(data_agg$nROI) / mean(data_agg$nROI), "\n")

# Check if the variance is roughly constant across nROI values (homoscedasticity)
model_lm_check <- lm(species_richness ~ nROI, data = data_agg)
par(mfrow = c(1, 2))
plot(model_lm_check, which = 1) # Residuals vs Fitted
plot(model_lm_check, which = 3) # Scale-Location plot
par(mfrow = c(1, 1))

#-----------------------------------------------------------------------------
# Model base: Baseline model with only the smooth fixed effect
#-----------------------------------------------------------------------------
# Minimal model with only nROI to compare
lm_base <- lm(
    species_richness ~ nROI,
    data = data_agg
)
# Check model summary and fixed effects p-values
summary(lm_base)

# Residual diagnostics using DHARMa
sim_res <- simulateResiduals(lm_base)
plot(sim_res) # Overall residual plot

# # OPTIONAL : More diagnostic graphs
# par(mfrow = c(3, 2)) # Set plotting layout
# # plot residuals vs fitted values. add title and red horizontal line at 0. change x and y labels
# plot(residuals(lm_base) ~ fitted(lm_base), main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
# abline(h = 0, lty = 2, col = "red")
# # plot residuals vs nROI. add title and red horizontal line at 0. change x and y labels
# plot(residuals(lm_base) ~ data_agg$nROI, main = "Residuals vs nROI", xlab = "nROI", ylab = "Residuals")
# abline(h = 0, lty = 2, col = "red")
# testDispersion(sim_res) # Test for overdispersion
# testZeroInflation(sim_res) # Test for zero-inflation
# testUniformity(sim_res) # Test for uniformity
# testOutliers(sim_res) # Test for outliers
# # Reset plotting layout
# par(mfrow = c(1, 1))

#-----------------------------------------------------------------------------
# model 1 Including fixed effects
#-----------------------------------------------------------------------------
# Fit the full model using the lm function (fixed effects only)
lm_model1 <- lm(
    species_richness ~ nROI + habitat + dataset,
    data = data_agg
)
# Check model summary and fixed effects p-values
summary(lm_model1)

# cat("The reference level for habitat is:", levels(data_agg$habitat)[1], "\n")
# cat("The reference level for device_id is:", levels(data_agg$device_id)[1], "\n")
# cat("The reference level for dataset is:", levels(data_agg$dataset)[1], "\n")

# Residual diagnostics using DHARMa
sim_res <- simulateResiduals(lm_model1)
plot(sim_res) # Overall residual plot

# # OPTIONAL : More diagnostic graphs
# par(mfrow = c(3, 2)) # Set plotting layout
# # plot residuals vs fitted values. add title and red horizontal line at 0. change x and y labels
# plot(residuals(lm_model1) ~ fitted(lm_model1), main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
# abline(h = 0, lty = 2, col = "red")
# # plot residuals vs nROI. add title and red horizontal line at 0. change x and y labels
# plot(residuals(lm_model1) ~ data_agg$nROI, main = "Residuals vs nROI", xlab = "nROI", ylab = "Residuals")
# abline(h = 0, lty = 2, col = "red")
# testDispersion(sim_res) # Test for overdispersion
# testZeroInflation(sim_res) # Test for zero-inflation
# testUniformity(sim_res) # Test for uniformity
# testOutliers(sim_res) # Test for outliers
# # Reset plotting layout
# par(mfrow = c(1, 1))

#-----------------------------------------------------------------------------
# model 2:  Including random effects
# glmmTMB is more flexible than lme4 and lme,
# glmmTMB works with levels never seen during model fitting (new levels)
# glmmTMB also allows 2 crossed random effects
#-----------------------------------------------------------------------------

# Fit the full model using the 'glmmTMB' function from glmmTMB
lmm_model2 <- glmmTMB(
    # species_richness ~ nROI + (1 | habitat) + (1 | site), # with cross random effects on habitat and site (site doesn't catch the spatial autocorrelation)
    # species_richness ~ nROI + (1 | habitat) + (1 | dataset), # with cross random effects on habitat and dataset (dataset catch the spatial autocorrelation) but structure in the residuals
    # species_richness ~ nROI + (nROI | habitat), # with random slope and intercept (not enough habitats to converge)
    species_richness ~ nROI + (1 | habitat), # with random intercept on habitat only
    data = data_agg,
    family = gaussian(link = "identity"), # Specifies a standard LMM (Gaussian)
    REML = TRUE # Uses REML, like lme by default for LMMs
)

# Check model summary and fixed effects p-values
summary(lmm_model2)

# Residual diagnostics using DHARMa
if (SAVE_FIGURES == TRUE) {
    png(filename = "./results/figure_S7.png", width = 20, height = 15, units = "cm", res = 300)
}

sim_res <- simulateResiduals(lmm_model2)
plot(sim_res) # Overall residual plot

dev.off()


# # # OPTIONAL : More diagnostic graphs
# par(mfrow = c(3, 2)) # Set plotting layout
# # plot residuals vs fitted values. add title and red horizontal line at 0. change x and y labels
# plot(residuals(lmm_model2) ~ fitted(lmm_model2), main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
# abline(h = 0, lty = 2, col = "red")
# # plot residuals vs nROI. add title and red horizontal line at 0. change x and y labels
# plot(residuals(lmm_model2) ~ data_agg$nROI, main = "Residuals vs nROI", xlab = "nROI", ylab = "Residuals")
# abline(h = 0, lty = 2, col = "red")
# testDispersion(sim_res) # Test for overdispersion
# testZeroInflation(sim_res) # Test for zero-inflation
# testUniformity(sim_res) # Test for uniformity
# testOutliers(sim_res) # Test for outliers
# # Reset plotting layout
# par(mfrow = c(1, 1))

#-----------------------------------------------------------------------------
# Build the graph with predictions, PI and CI
#-----------------------------------------------------------------------------

# Prediction plot with confidence and prediction intervals
pred_data <- data.frame(
    nROI = seq(min(data_agg$nROI), max(data_agg$nROI), length.out = 100)
)

# Get predictions with standard errors
pred_matrix <- predict(lmm_model2, newdata = pred_data, re.form = NA, se.fit = TRUE)
pred_data$predicted <- pred_matrix$fit
pred_data$se <- pred_matrix$se.fit

# Calculate degrees of freedom and critical t-value
n <- nrow(data_agg)
df_residual <- n - length(fixef(lmm_model2)$cond)
t_val <- qt(0.975, df_residual)

# Calculate residual standard error
sigma <- sigma(lmm_model2)

# Confidence intervals (uncertainty in the mean prediction)
pred_data$ci_lower <- pred_data$predicted - t_val * pred_data$se
pred_data$ci_upper <- pred_data$predicted + t_val * pred_data$se

# Prediction intervals (uncertainty for individual predictions)
pred_data$pi_se <- sqrt(pred_data$se^2 + sigma^2)
pred_data$pi_lower <- pred_data$predicted - t_val * pred_data$pi_se
pred_data$pi_upper <- pred_data$predicted + t_val * pred_data$pi_se

p_pred <- ggplot(data_agg, aes(x = nROI, y = species_richness)) +
    geom_ribbon(
        data = pred_data, aes(x = nROI, ymin = pi_lower,ymax = pi_upper),
        fill = "grey80", alpha = 0.5, inherit.aes = FALSE
    ) +
    geom_ribbon(
        data = pred_data, aes(x = nROI, ymin = ci_lower,ymax = ci_upper),
        fill = "blue", alpha = 0.3, inherit.aes = FALSE
    ) +
    geom_point(aes(color = habitat), alpha = 0.75, size = 3) +
    geom_line(
        data = pred_data, aes(x = nROI, y = predicted),
        color = "blue", size = 1
    ) +
    geom_line(
        data = pred_data, aes(x = nROI, y = pi_lower),
        color = "grey50", linetype = "dashed", size = 0.8
    ) +
    geom_line(
        data = pred_data, aes(x = nROI, y = pi_upper),
        color = "grey50", linetype = "dashed", size = 0.8
    ) +
    theme_minimal() +
    coord_cartesian(xlim = c(0, 260), ylim = c(0, 8)) +
    theme(legend.position = c(0.82, 0.17))
print(p_pred)
par(mfrow = c(1, 1))

# save the plot
if (SAVE_FIGURES == TRUE) {
    ggsave("./results/figure_S8.png", plot = p_pred, dpi = 300)
}

#-------------------------------------------------------------
# metrics
#-------------------------------------------------------------

# calculate R2 for mixed models with random effects
r2_values <- r2(lmm_model2)
cat("Marginal R² (fixed effects):", r2_values$R2_marginal, "\n")
cat("Conditional R² (fixed + random effects):", r2_values$R2_conditional, "\n")

# add RMSE, MAE and Bias calculation
predicted_values <- predict(lmm_model2)
residuals_values <- data_agg$species_richness - predicted_values
rmse <- sqrt(mean(residuals_values^2))
cat("RMSE of the model:", rmse, "\n")
mae <- mean(abs(residuals_values))
cat("MAE of the model:", mae, "\n")
bias <- mean(residuals_values)
cat("Bias of the model:", bias, "\n")
r2 <- 1 - sum(residuals_values^2) / sum((data_agg$species_richness - mean(data_agg$species_richness))^2)
cat("Residual R² of the model:", r2, "\n")

#  equation of the model when using only nROI as fixed effect
fixed_effects <- summary(lmm_model2)$coefficients$cond
slope <- fixed_effects["nROI", "Estimate"]
intercept <- fixed_effects["(Intercept)", "Estimate"]
cat("Model equation: species_richness =", slope, "* nROI +", intercept, "\n")

# test model on a new dataset: WABAD
# =============================================================================

# min sampling effort
SAMPLING_EFFORT <- 3

# Load new dataset
new_data_path <- "./results/test_dataset_for_statistical_modeling_in_R.csv"
new_data <- read.csv(new_data_path)

# remove sites with count less than SAMPLING_EFFORT
new_data <- new_data[new_data["count"] >= SAMPLING_EFFORT, ]

# set habitat and dataset as factors
new_data$habitat <- as.factor(new_data$habitat)

# rename column site into dataset
colnames(new_data)[colnames(new_data) == "site"] <- "dataset"
new_data$dataset <- as.factor(new_data$dataset)

# Make predictions on the new dataset using nROI
# Exclude random effects to avoid factor level mismatch
# Ensure habitat levels match the training data
new_data$habitat <- factor(new_data$habitat, levels = levels(data_agg$habitat))
new_data$dataset <- factor(new_data$dataset, levels = levels(data_agg$dataset))

# do prediction with the best mixed-effects model
predictions <- predict(lmm_model2,
    newdata = new_data,
    re.form = NA
)

# Add predictions to the new dataset
new_data$predicted_species_richness <- predictions
# View the new dataset with predictions
head(new_data)

# compare predicted species_richness with observed species_richness.
p_pred <- ggplot(new_data, aes(x = predicted_species_richness, y = species_richness)) +
    geom_point(color = "blue", alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    coord_fixed(
        xlim = c(0, 14), ylim = c(0, 14), ratio = 1 # Force le ratio 1:1 pour un graphique carré
    )
labs(
    title = "Predicted vs Observed Species Richness (WABAD dataset)",
    x = "Predicted Species Richness",
    y = "Observed Species Richness"
) +
    theme_minimal()
print(p_pred)

# compute RMSE, MAE, bias, R2, CCC between predicted and observed species_richness
residuals_values <- new_data$predicted_species_richness - new_data$species_richness
rmse <- sqrt(mean(residuals_values^2))
mae <- mean(abs(residuals_values))
bias <- mean(residuals_values)
r2 <- 1 - sum(residuals_values^2) / sum((new_data$species_richness - mean(new_data$species_richness))^2)
ccc <- CCC(new_data$predicted_species_richness, new_data$species_richness)

# print the results
cat("Model performance on WABAD dataset:\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("Bias:", bias, "\n")
cat("R²:", r2, "\n")
cat("CCC:", ccc$rho.c$est, "\n")

# =============================================================================

# Spatial autocorrelation analysis (if spatial coordinates are available)

# =============================================================================

# Load required spatial packages
if (!require("spdep")) install.packages("spdep")
if (!require("sp")) install.packages("sp")
if (!require("sf")) install.packages("sf")
library(spdep) # Spatial dependence analysis
library(sp) # Spatial data classes
library(sf) # Simple features for spatial data

###### SELECT THE MODEL RESIDUALS TO TEST FOR SPATIAL AUTOCORRELATION #######

# OPTION 1 : Here we use the residuals from the k-fold robust linear regression model
# (see 03_indices_resgression_linear_Groupkfold_by_habitat.ipynb)
#------------------------------------------------------------------------
# # Extract the residuals from the csv file residuals_simple_regression.csv
# # Load the data
# residual_path <- "./results/nROI_residuals_simple_regression.csv"
# if (!file.exists(residual_path)) {
#     stop("Data file not found. Please check the path: ", residual_path)
# }
# model_residuals <- read.csv(residual_path, stringsAsFactors = FALSE)

# # remove rows with habitat that are not in data_agg
# model_residuals <- model_residuals[model_residuals$site %in% levels(data_agg$site), ]

# # add a column residual in the dataframe data_agg
# data_agg$residual <- model_residuals$residual

# OPTION 2: Or you can directly extract the residuals from the GLMM model (lm_model1 lmm_model2)
#------------------------------------------------------------------------
# Extract the residuals from the model
data_agg$residual <- residuals(lmm_model2) # or lm_base lm_model1 lmm_model2

# ##############################################################################

# Add the residuals and coordinates to a new data frame
residuals_data <- data.frame(
    residual = data_agg$residual,
    longitude = data_agg$LON,
    latitude = data_agg$LAT
)

# Remove NA (sites that were not used for the regression) from residuals_data
residuals_data <- residuals_data[complete.cases(residuals_data[c("residual", "longitude", "latitude")]), ]

# Convert the residuals data frame to a spatial object
spatial_residuals <- st_as_sf(residuals_data, coords = c("longitude", "latitude"), crs = 4326)

# Extract coordinates for the weights matrix
coords <- st_coordinates(spatial_residuals)

# Create a k-nearest neighbors spatial weights matrix
neighbors_list <- knearneigh(coords, k = 4)
spatial_weights <- nb2listw(knn2nb(neighbors_list), style = "W")

# Perform the Moran's I test on the residuals
moran_test_residuals <- moran.test(spatial_residuals$residual, listw = spatial_weights)

print(moran_test_residuals)

# Interpretation of Moran's I test results
if (moran_test_residuals$p.value < 0.05) {
    if (moran_test_residuals$estimate["Moran I statistic"] > 0) {
        cat("Significant positive spatial autocorrelation in residuals detected!\n")
        cat("This indicates the model may be missing spatial predictors or structure.\n")
        cat("Consider adding spatial terms to your model.\n")
    } else {
        cat("Significant negative spatial autocorrelation in residuals detected!\n")
        cat("This is unusual and may indicate model misspecification.\n")
    }
} else {
    cat("No significant spatial autocorrelation in residuals.\n")
    cat("The model adequately accounts for spatial structure.\n")
}
