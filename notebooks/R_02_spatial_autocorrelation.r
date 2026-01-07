# =============================================================================
# Statistical Modeling with Generalized Linear Mixed Models (GLMM) in R
# Translation from Python code with proper random effects implementation
# Updated: January 2025
# =============================================================================

# Clear workspace
rm(list = ls())

# set wd to the script location
setwd("./")

# # Load required libraries
# # =============================================================================
# Load required spatial packages
if (!require("dplyr")) install.packages("dplyr")
if (!require("spdep")) install.packages("spdep")
if (!require("sp")) install.packages("sp")
if (!require("sf")) install.packages("sf")
library(dplyr) # Data manipulation
library(spdep) # Spatial dependence analysis
library(sp) # Spatial data classes
library(sf) # Simple features for spatial data

# File paths
# # ===========================================================================
FILENAME_DATA <- "./results/data_statistical_model.csv"
FILENAME_RESIDUALS <- "./results/nROI_residuals_simple_regression.csv"

# Data loading and preparation
# =============================================================================
cat("Loading and preparing data...\n")

# Load the data
data_path <- FILENAME_DATA
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
    group_by(site, habitat, device_id, dataset, LAT, LON) %>%
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

# =============================================================================

# Spatial autocorrelation analysis (if spatial coordinates are available)

# =============================================================================

cat("SPATIAL AUTOCORRELATION ANALYSIS\n")

cat("Required GPS coordinate format:\n")
cat("- Latitude: Decimal degrees (e.g., 45.7640)\n")
cat("- Longitude: Decimal degrees (e.g., 4.8357)\n")
cat("- Coordinate system: WGS84 (EPSG:4326) recommended\n")
cat("- Column names: 'latitude' and 'longitude' or 'lat' and 'lon'\n\n")

# Extract the residuals from the csv file residuals_simple_regression.csv
# Load the data
residual_path <- FILENAME_RESIDUALS
if (!file.exists(residual_path)) {
    stop("Data file not found. Please check the path: ", residual_path)
}
model_residuals <- read.csv(residual_path, stringsAsFactors = FALSE)

# add a column residual in the dataframe data_agg
data_agg$residual <- model_residuals$residual

# Remove NA (sites that were not used for the regression) from residuals_data
data_agg_clean <- data_agg[complete.cases(data_agg), ]

# Add the residuals and coordinates to a new data frame
residuals_data <- data.frame(
    residual = data_agg_clean$residual,
    longitude = data_agg_clean$LON,
    latitude = data_agg_clean$LAT
)

# Convert the residuals data frame to a spatial object
spatial_residuals <- st_as_sf(residuals_data, 
                            coords = c("longitude", "latitude"), 
                            crs = 4326)

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

# find the correlated sites
moran_loc <- localmoran(spatial_residuals$residual, listw = spatial_weights)
# add the results to the spatial_residuals dataframe
spatial_residuals$moran_I <- moran_loc[, "Ii"]
# add the p-value to the spatial_residuals dataframe
spatial_residuals$p_value <- moran_loc[, "Pr(z != E(Ii))"]
# print all sites with a significant positive spatial autocorrelation
cat("\nSites with significant positive spatial autocorrelation (p < 0.05):\n")
print(data_agg_clean[spatial_residuals$p_value < 0.05 & spatial_residuals$moran_I > 0, ], n=50)

# remove the sites with a significant positive spatial autocorrelation from the data_agg dataframe
sites <- data_agg_clean[spatial_residuals$p_value < 0.05 & spatial_residuals$moran_I > 0, ]
print(sites$site)
