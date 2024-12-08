from skimage.morphology import opening, dilation, closing
import numpy as np
import pandas as pd  # for csv
import matplotlib.pyplot as plt
from maad import sound, features, util, rois
from scipy.stats import spearmanr
from datetime import datetime
from math import floor
from numpy import mean
from numpy import std
from numpy import absolute
import warnings

## scikit-learn (machine learning) package
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, permutation_test_score

# umap
import umap 

# HDBSCAN package (clustering)
import hdbscan

## 

MIN_V = 1 # min week number possible
MIN_m = 1 # min month number possible
MIN_d = 1 # min day number possible
MIN_H = 0 # min hour number possible
MIN_M = 0 # min minute number possible

MAX_V = 53 # max week number possible
MAX_m = 12 # max month number possible
MAX_d = 31 # max day number possible
MAX_H = 23 # max hour number possible
MAX_M = 59 # max minute number possible



###############################################################################
def _prepare_features(df_features,
                    scaler = "STANDARDSCALER",
                    features = ["shp", "centroid_f"]):
    """

    Prepare the features before clustering

    Parameters
    ----------
    df_features : pandas dataframe
        the dataframe should contain the features 
    scaler : string, optional {"STANDARDSCALER", "ROBUSTSCALER", "MINMAXSCALER"}
        Select the type of scaler uses to normalize the features.
        The default is "STANDARDSCALER".
    features : list of features, optional
        List of features will be used for the clustering. The name of the features
        should be the name of a column in the dataframe. In case of "shp", "shp"
        means that all the shpxx will be used.
        The default is ["shp","centroid_f"].

    Returns
    -------
    X : pandas dataframe
        the dataframe with the normalized features 

    """

    # select the scaler
    #----------------------------------------------
    if scaler == "STANDARDSCALER":
        scaler = StandardScaler() #
    elif scaler == "ROBUSTSCALER":
        scaler = RobustScaler()
    elif scaler == "MINMAXSCALER" :
        scaler = MinMaxScaler() 
    else :
        scaler = StandardScaler()
        print ("*** WARNING *** the scaler {} does not exist. StandarScaler was choosen".format(scaler))
    
    X = pd.DataFrame()
    f = features.copy() #copy the list otherwise the suppression of a feature is kept
    
    # Normalize the shapes
    #----------------------------------------------
    if "shp" in f :
        # with shapes
        # create a vector with X in order to scale all features together
        X = df_features.loc[:, df_features.columns.str.startswith("shp")]
        X_vect = X.to_numpy()
        X_shape = X_vect.shape
        X_vect = X_vect.reshape(X_vect.size, -1)
        X_vect = scaler.fit_transform(X_vect)
        X = X_vect.reshape(X_shape)
        # remove "shp" from the list
        f.remove('shp')
    
    X2 = pd.DataFrame()
    
    # Normalize the other features (centroid, bandwidth...)
    #-------------------------------------------------------
    # test if the features list is not null
    if len(f) > 0 :
        # add other features like frequency centroid
        X2 = df_features[f]
        # Preprocess data : data scaler
        X2 = scaler.fit_transform(X2)

    # Concatenate the features after normalization
    #-------------------------------------------------------
    if len(X) == len(X2)  :
        # create a matrix with all features after rescaling
        X = np.concatenate((X, X2), axis=1)
    elif len(X2) > len(X) :
        X2 = X
    
    return X

#=============================================================================

def region_of_interest_index2(Sxx_power, tn, fn, 
                            seed_level=16, 
                            low_level=6, 
                            fusion_rois=(0.05, 100),
                            remove_rois_fmin_lim = 50,
                            remove_rois_fmax_lim = None,
                            remove_rain = True,
                            min_event_duration=None, 
                            max_event_duration=None, 
                            min_freq_bw=None, 
                            max_freq_bw=None, 
                            max_xy_ratio=None,
                            get_clusters=False,
                            display=False,
                            verbose=False,
                            **kwargs):

    """
    Calculates region of interest (ROI) indices based on a spectrogram.

    Parameters:
    - Sxx_power (numpy.ndarray): spectrogram with power values (Sxx_amplitude²).
    - tn (numpy.ndarray): Time axis values.
    - fn (numpy.ndarray): Frequency axis values.
    - seed_level (int): Parameter for binarization of the spectrogram (default: 16).
    - low_level (int): Parameter for binarization of the spectrogram (default: 6).
    - fusion_rois (tuple): Fusion parameters for ROIs (default: (0.05, 100)).
    - remove_rois_fmin_lim (float or tuple): Frequency threshold(s) to remove ROIs (default: None).
    - remove_rois_fmax_lim (float): Frequency threshold to remove ROIs (default: None).
    - remove_rain (bool): Whether to remove rain noise (default: True).
    - min_event_duration (float): Minimum time duration of an event in seconds (default: None).
    - max_event_duration (float): Maximum time duration of an event in seconds (default: None).
    - min_freq_bw (float): Minimum frequency bandwidth in Hz (default: None).
    - max_freq_bw (float): Maximum frequency bandwidth in Hz (default: None).
    - display (bool): Whether to display intermediate results (default: False).
    - **kwargs: Additional arguments.

    Returns:
    - nROI (int): Number of ROIs.
    - aROI (float): Percentage of ROIs area over the total area.
    - df_rois (pandas.DataFrame): DataFrame containing information about the ROIs.
    - df_clusters (pandas.DataFrame): DataFrame containing information about the clusters.

    Note:
    - This function requires the following packages: skimage, numpy, pandas, matplotlib, maad.

    """

    FUSION_ROIS = fusion_rois
    REMOVE_ROIS_FMIN_LIM = remove_rois_fmin_lim
    REMOVE_ROIS_FMAX_LIM = remove_rois_fmax_lim
    REMOVE_RAIN = remove_rain
    MIN_EVENT_DUR = min_event_duration # Minimum time duration of an event (in s)
    MAX_EVENT_DUR = max_event_duration
    MIN_FREQ_BW = min_freq_bw # Minimum frequency bandwidth (in Hz)
    MAX_FREQ_BW = max_freq_bw
    if (MIN_EVENT_DUR is not None) and (MIN_FREQ_BW is not None):  
        MIN_ROI_AREA = MIN_EVENT_DUR * MIN_FREQ_BW 
    else :
        MIN_ROI_AREA = None
    if (MAX_EVENT_DUR is not None) and (MAX_FREQ_BW is not None):  
        MAX_ROI_AREA = MAX_EVENT_DUR * MAX_FREQ_BW 
    else :
        MAX_ROI_AREA = None
    
    MAX_XY_RATIO = max_xy_ratio
    
    BIN_H = seed_level
    BIN_L = low_level

    GET_CLUSTERS = get_clusters
    
    DISPLAY = display
    EXT = kwargs.pop('ext',(tn[0], tn[-1], fn[0], fn[-1])) 

    """*********************** Convert into dB********* ********************"""
    #
    Sxx_dB = util.power2dB(Sxx_power, db_range=96) + 96

    """*********************** Remove stationnary noise ********************"""       
    #### Use median_equalizer function as it is fast reliable
    if REMOVE_RAIN: 
        # remove single vertical lines
        Sxx_dB_without_rain, _ = sound.remove_background_along_axis(Sxx_dB.T,
                                                            mode='mean',
                                                            N=1,
                                                            display=False)
        Sxx_dB = Sxx_dB_without_rain.T

    # ==================================================================================> TO COMMENT
    # remove single horizontal lines
    Sxx_dB_noNoise, _ = sound.remove_background_along_axis(Sxx_dB,
                                                    mode='median',
                                                    N=1,
                                                    display=False)
    # ==================================================================================> END TO COMMENT (change)

    Sxx_dB_noNoise[Sxx_dB_noNoise<=0] = 0
    
    """**************************** Find ROIS ******************************"""  
    # Time resolution (in s)
    DELTA_T = tn[1]-tn[0]
    # Frequency resolution (in Hz)
    DELTA_F = fn[1]-fn[0]

    # snr estimation to threshold the spectrogram
    _,bgn,snr,_,_,_ = sound.spectral_snr(util.dB2power(Sxx_dB_noNoise))
    if verbose :
        print('BGN {}dB / SNR {}dB'.format(bgn,snr))
        

    # binarization of the spectrogram to select part of the spectrogram with 
    # acoustic activity
    im_mask = rois.create_mask(
        Sxx_dB_noNoise,  
        mode_bin = 'absolute', 
        bin_h=BIN_H, 
        bin_l=BIN_L,
        # bin_h=snr+BIN_H,
        # bin_l=snr+BIN_L,
        # bin_h=util.add_dB(BIN_H,snr),
        # bin_l=util.add_dB(BIN_L,snr),
        )    
    
    if verbose :
        print('bin_h {}dB / bin_l {}dB'.format(util.add_dB(BIN_H,snr),util.add_dB(BIN_L,snr)))
        
    """**************************** Fusion ROIS ******************************"""  
    if type(FUSION_ROIS) is tuple :
        Ny_elements = round(FUSION_ROIS[0] / DELTA_T)
        Nx_elements = round(FUSION_ROIS[1] / DELTA_F)
        im_mask = closing(im_mask, footprint=np.ones([Nx_elements,Ny_elements]))

    # get the mask with rois (im_rois) and the bounding box for each rois (rois_bbox) 
    # and an unique index for each rois => in the pandas dataframe rois
    im_rois, df_rois  = rois.select_rois(
        im_mask,
        min_roi=MIN_ROI_AREA, 
        max_roi=MAX_ROI_AREA)
    
    """**************************** add a column ratio_xy ****************************"""  
    # add ratio x/y
    df_rois['ratio_xy'] = (df_rois.max_y -df_rois.min_y) / (df_rois.max_x -df_rois.min_x) 

    """************ remove some ROIs based on duration and bandwidth *****************"""  
    # remove min and max duration 
    df_rois['duration'] = (df_rois.max_x -df_rois.min_x) * DELTA_T 
    if MIN_EVENT_DUR is not None :
        df_rois = df_rois[df_rois['duration'] >= MIN_EVENT_DUR]
    if MAX_EVENT_DUR is not None :    
        df_rois = df_rois[df_rois['duration'] <= MAX_EVENT_DUR]
    df_rois.drop(columns=['duration'])

    # remove min and max frequency bandwidth 
    df_rois['bw'] = (df_rois.max_y -df_rois.min_y) * DELTA_F 
    if MIN_FREQ_BW is not None :
        df_rois = df_rois[df_rois['bw'] >= MIN_FREQ_BW]
    if MAX_FREQ_BW is not None :    
        df_rois = df_rois[df_rois['bw'] <= MAX_FREQ_BW]
    df_rois.drop(columns=['bw'])
    

    """**************************** Remove some ROIS ******************************"""  
    if len(df_rois) >0 :
        if REMOVE_ROIS_FMIN_LIM is not None:
            low_frequency_threshold_in_pixels=None
            high_frequency_threshold_in_pixels=None

            if isinstance(REMOVE_ROIS_FMIN_LIM, (float, int)) :
                low_frequency_threshold_in_pixels = max(1, round(REMOVE_ROIS_FMIN_LIM / DELTA_F))
            elif isinstance(REMOVE_ROIS_FMIN_LIM, (tuple, list, np.ndarray)) and len(REMOVE_ROIS_FMIN_LIM) == 2 :
                low_frequency_threshold_in_pixels = max(1, round(REMOVE_ROIS_FMIN_LIM[0] / DELTA_F))
                high_frequency_threshold_in_pixels = min(im_rois.shape[1]-1, round(REMOVE_ROIS_FMIN_LIM[1] / DELTA_F))
            else:
                raise ValueError ('REMOVE_ROIS_FMAX_LIM should be {None, a single value or a tuple of 2 values')

            # retrieve the list of labels that match the condition
            list_labelID = df_rois[df_rois['min_y']<=low_frequency_threshold_in_pixels]['labelID']
            # set to 0 all the pixel that match the labelID that we want to remove
            for labelID in list_labelID.astype(int).tolist() :
                im_rois[im_rois==labelID] = 0
            # delete the rois corresponding to the labelID that we removed in im_mask
            df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]

            if high_frequency_threshold_in_pixels is not None :
                # retrieve the list of labels that match the condition  
                list_labelID = df_rois[df_rois['min_y']>=high_frequency_threshold_in_pixels]['labelID']
                # set to 0 all the pixel that match the labelID that we want to remove
                for labelID in list_labelID.astype(int).tolist() :
                    im_rois[im_rois==labelID] = 0
                # delete the rois corresponding to the labelID that we removed in im_mask
                df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]

        if REMOVE_ROIS_FMAX_LIM is not None:
            if isinstance(REMOVE_ROIS_FMAX_LIM, (float, int)) :
                high_frequency_threshold_in_pixels = min(im_rois.shape[1]-1, round(REMOVE_ROIS_FMAX_LIM / DELTA_F))
            else:
                raise ValueError ('REMOVE_ROIS_FMAX_LIM should be {None, or single value')

            # retrieve the list of labels that match the condition  
            list_labelID = df_rois[df_rois['max_y']>=high_frequency_threshold_in_pixels]['labelID']
            # set to 0 all the pixel that match the labelID that we want to remove
            for labelID in list_labelID.astype(int).tolist() :
                im_rois[im_rois==labelID] = 0
            # delete the rois corresponding to the labelID that we removed in im_mask
            df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]
        
        if MAX_XY_RATIO is not None:
            df_rois = df_rois[df_rois['ratio_xy'] < MAX_XY_RATIO]  

    """**************************** Index calculation ******************************""" 

    # Convert boolean (True or False) into integer (0 or 1)
    im_mask_filtered = im_rois>0 * 1
    # number of ROIs / min
    ROI_number = len(df_rois) / (tn[-1] / 60) 
    # ROIs coverage in %
    ROI_cover = im_mask_filtered.sum() / (im_mask_filtered.shape[0]*im_mask_filtered.shape[1]) *100 

    if verbose :
        print('===> ROI_number : {:.0f}#/min | ROI_cover : {:.2f}%'.format(round(ROI_number), ROI_cover))

    """****************************************************************************""" 
    """******************************* CLUSTERING *********************************""" 
    """****************************************************************************""" 
    if GET_CLUSTERS:

        # try:

        if len(df_rois) > 2 :
            """**************************** Compute features ******************************""" 
            # keep only the portion of the spectrogram corresponding on the mask
            # ==================================================================================> TO COMMENT
            Sxx_dB_noNoise_filtered = Sxx_dB_noNoise * im_mask_filtered
            # Sxx_dB_noNoise_filtered = Sxx_dB_noNoise 
            # ==================================================================================> END TO COMMENT

            df_rois = util.format_features(df_rois, tn, fn)
            df_shape, params_shape = features.shape_features(Sxx_dB_noNoise_filtered, 
                                                            resolution='low',
                                                            rois=df_rois)
            df_shape = util.format_features(df_shape, tn, fn)

            # create a new dataframe with features
            df_features = df_shape.copy()

            # frequency mean
            df_features['peak_f'] = (df_rois['max_f'] + df_rois['min_f']) / 2

            # # duration
            # df_features['duration_t'] = (df_rois['max_t'] - df_rois['min_t']) 

            # drop NaN rows
            df_features = df_features.dropna(axis=0)

            """**************************** find clusters ******************************""" 

            # Prepare the features of that categories
            #--------------------------------------------------------------------

            # ! Normalizing the features for each audio file will cause different values dispersion around the median values which make it difficult to find
            # a universal EPS
            # X = _prepare_features(df_features, scaler ='MINMAXSCALER', features = ['shp', 'centroid_f', 'peak_f', 'duration_t']) # STANDARDSCALER ROBUSTSCALER MINMAXSCALER
            # X = _prepare_features(df_features, scaler ='MINMAXSCALER', features = ['shp', 'centroid_f', 'peak_f']) # STANDARDSCALER ROBUSTSCALER MINMAXSCALER
            # X = _prepare_features(df_features, scaler ='MINMAXSCALER', features = ['shp', 'centroid_f']) # STANDARDSCALER ROBUSTSCALER MINMAXSCALER
            # ==================================================================================> TO COMMENT
            X = _prepare_features(df_features, scaler ='MINMAXSCALER', features = ['shp', 'peak_f']) # STANDARDSCALER ROBUSTSCALER MINMAXSCALER
            # ==================================================================================> END TO COMMENT

            # ==================================================================================> TO COMMENT
            # => GIVE ABOUT THE SAME RESULTS AS WITH _prepare_features
            # X  = df_features.loc[:, df_features.columns.str.startswith("shp")].to_numpy()
            # X2 = df_features['peak_f'].to_numpy() / df_features['peak_f'].median()
            # # Add a new axis to array2
            # X2 = np.expand_dims(X2, axis=1)
            # X  = np.concatenate((X, X2), axis=1)
            # ==================================================================================> END TO COMMENT

            # PCA dimensionality reduction : 2 components
            #---------------------------------------------------------------------
            
            N_COMPONENTS = 2
            X = PCA(n_components=N_COMPONENTS).fit_transform(X)

            if DISPLAY :
                fig = plt.figure()
                ax = fig.add_subplot(121)
                ax.scatter(X[:,0], X[:,1])
        
            # Estimating the EPS by finding the knee (or elbow)
            #---------------------------------------------------------------------

            # Calculate the average distance between each point in the data set and
            # its N nearest neighbors (N corresponds to min_points).
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=2)
            neighbors_fit = neighbors.fit(X)
            distances, indices = neighbors_fit.kneighbors(X)
            # Sort distance values by ascending value and plot
            distances = np.sort(distances, axis=0)
            distances = distances[:, 1] 
            # Kneed package to find the knee of a curve
            from kneed import KneeLocator
            # Filter out warnings from the specific function
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # find the knee (curvature inflexion point)
                kneedle = KneeLocator(
                    x=np.arange(0, len(distances), 1),
                    y=distances,
                    interp_method="polynomial", # interp1d polynomial
                    curve="convex",
                    direction="increasing",
                )       
            # first find the maximum distance that corresponds to 95% of observations
            try :
                EPS = float(kneedle.knee_y)  
            except :
                EPS = distances [-1]
            
            if verbose :
                print("=> EPS {:.3f}".format(EPS))

            # Hierachical density clustering with HDBSCAN
            #---------------------------------------------------------------------
            # ==================================================================================> TO CHANGE 1 or 0.5
            MIN_POINTS = max(2, round(len(df_features) * 1/100))
            MIN_SAMPLES = MIN_POINTS #2 # max(1, MIN_POINTS - 1)
            
            if verbose :
                print("=> MIN_POINTS {} | MIN_SAMPLES {}".format(MIN_POINTS, MIN_SAMPLES))

            try :
                cluster = hdbscan.HDBSCAN(
                                    min_cluster_size=MIN_POINTS,
                                    min_samples=MIN_SAMPLES,
                                    cluster_selection_epsilon = max(0,EPS*1), 
                                    core_dist_n_jobs = -1,
                                ).fit(X)
                
                if DISPLAY :
                    ax2 = fig.add_subplot(122)
                    ax2.scatter(X[:,0], X[:,1], c = cluster.labels_)

                # # add the cluster number and the label found with the clustering
                # #---------------------------------------------------------------
                df_clusters = df_rois.copy()
                # add the cluster number into the label's column of the dataframe
                df_clusters["label"] = cluster.labels_.reshape(-1, 1)
                # convert the cluster number into integer
                df_clusters['label'] = df_clusters['label'].astype('int')

                # # convert the label -1 (noise) into unique label as they might be single songs/calls
                # highest_number = df_clusters["label"].max()
                # mask = df_clusters["label"] == -1
                # new_labels = range(highest_number + 1, highest_number + 1 + mask.sum())
                # df_clusters.loc[mask, "label"] = new_labels

            except:
                df_clusters = df_rois.copy()
                df_clusters["label"] = -1

            """**************************** acoustics indicators ******************************""" 
            # remove label -1
            filtered_df = df_clusters[df_clusters['label'] != -1]['label']

            # if there is at least one cluster 
            if len(filtered_df) >0 : 
                # number of unique clusters
                ROI_nb_clusters = filtered_df.nunique()

                # abundances
                total_non_minus_one_labels = len(filtered_df)
                label_counts = filtered_df.value_counts()
                relative_abundance = label_counts / total_non_minus_one_labels

                # shannon
                ROI_shannon =-sum(relative_abundance*np.log(relative_abundance))

                # simpson
                ROI_simpson = sum(relative_abundance**2)

                # inverse simpson
                ROI_inv_simpson = 1 / ROI_simpson

                # gini simpson
                ROI_gini_simpson = 1 - ROI_simpson
            
            else :
                ROI_nb_clusters = 0
                ROI_shannon = None
                ROI_simpson = None
                ROI_inv_simpson = 0
                ROI_gini_simpson = None
        else :
            ROI_nb_clusters = 0
            ROI_shannon = None
            ROI_simpson = None
            ROI_inv_simpson = 0
            ROI_gini_simpson = None
            df_clusters = df_rois
            df_clusters['label'] = ''
            df_features = None  
        
        # except :
        #     ROI_nb_clusters = None
        #     ROI_shannon = None
        #     ROI_simpson = None
        #     ROI_inv_simpson = None
        #     ROI_gini_simpson = None
        #     df_clusters = df_rois
        #     df_clusters['label'] = ''
        #     df_features = None  

        """****************************************************************************""" 
        """**************************** if NO CLUSTERING ******************************""" 
        """****************************************************************************""" 
    else :
        ROI_nb_clusters = None
        ROI_shannon = None
        ROI_simpson = None
        ROI_inv_simpson = None
        ROI_gini_simpson = None
        df_clusters = df_rois
        df_clusters['label'] = ''
        df_features = None

        """**************************** Display process ******************************"""

        
    LABELS = list(df_clusters['label'].unique()).sort()

    if GET_CLUSTERS :
        COLORS = ['tab:red', 'tab:green', 'tab:orange', 'tab:blue', 
                'tab:purple','tab:pink','tab:brown','tab:olive',
                'tab:cyan','tab:gray','yellowgreen','plum', 'green','darkcyan']
    else :
        COLORS = ['yellow']

    if DISPLAY : 
        fig, ax = plt.subplots(4,1,figsize=(6/10*tn[-1],8), sharex=True)
        # util.plot_wave(wave, fs,  xlabel='', ax=ax[0])
        # util.plot_wave(env, fs, ax=ax[1])
        # util.plot_spectrum(pxx, fidx, ax=ax[2])
        util.plot_spectrogram(Sxx_power, extent=EXT, ax=ax[0], ylabel ='', xlabel='', colorbar=False)
        util.plot_spectrogram(Sxx_dB_noNoise, log_scale=False, vmax = np.percentile(Sxx_dB_noNoise,99.9), vmin = np.percentile(Sxx_dB_noNoise,0.1), ylabel ='', xlabel='', extent=EXT, ax=ax[1], colorbar=False)
        # util.plot_spectrogram(Sxx_dB_noNoise_norain, log_scale=False, vmax = np.percentile(Sxx_dB_noNoise_norain,99.9), vmin = np.percentile(Sxx_dB_noNoise_norain,0.1), xlabel='', extent=EXT, ax=ax[3], colorbar=False)
        # util.plot2d(im_mask, ylabel ='', xlabel='', extent=EXT, ax=ax[3], colorbar=False)
        util.plot2d(im_mask, ylabel ='', xlabel='', extent=EXT, ax=ax[2], colorbar=False)
        # util.plot2d(im_rois, ylabel ='', xlabel='', extent=EXT, cmap = util.rand_cmap(len(df_rois), seed=1), interpolation = 'none', ax=ax[5], colorbar=False)
        util.plot_spectrogram(Sxx_power, ylabel ='', extent=EXT, ax=ax[3], colorbar=False)
        if len(df_clusters) > 0 :
            util.overlay_rois(im_ref=Sxx_power, 
                        rois = df_clusters,
                        ylabel ='', 
                        extent=EXT, 
                        ax=ax[3], 
                        colorbar=False,
                        unique_labels=LABELS,
                        edge_color=COLORS,
                        textbox_label=True)

            
    return ROI_number, ROI_cover, ROI_nb_clusters, ROI_shannon, ROI_simpson, ROI_inv_simpson, ROI_gini_simpson, df_features, df_clusters 


def date_from_filename (filename):
    """
    Extract date and time from the filename. Return a datetime object
    
    Parameters
    ----------
    filename : string
    The filename must follow this format :
    XXXX_yyyymmdd_hhmmss.wav
    with yyyy : year / mm : month / dd: day / hh : hour (24hours) /
    mm : minutes / ss : seconds
            
    Returns
    -------
    date : object datetime
        This object contains the date of creation of the file extracted from
        the filename postfix. 
    """
    # date by default
    date = datetime(1900,1,1,0,0,0,0)
    # test if it is possible to extract the recording date from the filename
    if filename[-19:-15].isdigit(): 
        yy=int(filename[-19:-15])
    else:
        return date
    if filename[-15:-13].isdigit(): 
        mm=int(filename[-15:-13])
    else:
        return date
    if filename[-13:-11].isdigit(): 
        dd=int(filename[-13:-11])
    else:
        return date
    if filename[-10:-8].isdigit(): 
        HH=int(filename[-10:-8])
    else:
        return date
    if filename[-8:-6].isdigit(): 
        MM=int(filename[-8:-6])
    else:
        return date
    if filename[-6:-4].isdigit(): 
        SS=int(filename[-6:-4])
    else:
        return date

    # extract date and time from the filename
    date = datetime(year=yy, month=mm, day=dd, hour=HH, minute=MM, second=SS, 
                    microsecond=0)
    
    return date


###################
# adapted from : https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/
class RobustRegressor:
    def __init__(self, regressor_type='LinearRegression', *args, **kwargs):

        self._scores = []

        if regressor_type == 'LinearRegression':
            self.regressor = LinearRegression(*args, **kwargs)
        elif regressor_type == 'TheilSenRegressor':
            self.regressor = TheilSenRegressor(*args, **kwargs)
        elif regressor_type == 'RANSACRegressor':
            self.regressor = RANSACRegressor(*args, **kwargs)
        elif regressor_type == 'HuberRegressor':
            self.regressor = HuberRegressor(*args, **kwargs)
        else:
            raise ValueError("Invalid regressor type. Choose from 'LinearRegression', 'TheilSenRegressor', 'RANSACRegressor', or 'HuberRegressor'.")

    def fit(self, X, y, random_state=None, n_jobs=None, *args, **kwargs):

        N_SPLITS = kwargs.pop('n_splits',2) 
        N_REPEATS = kwargs.pop('n_repeats',50) 

        cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=random_state)

        R_SQUARED = absolute(cross_val_score(self.regressor, X, y, scoring='r2', cv=cv, n_jobs=n_jobs))  
        MAE = absolute(cross_val_score(self.regressor, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=n_jobs))  
        RMSE = absolute(cross_val_score(self.regressor, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=n_jobs))  

        # Evaluate the significance of a cross-validated score with permutations.
        # Permutes targets to generate ‘randomized data’ and compute the empirical p-value 
        # against the null hypothesis that features and targets are independent.
        _, _, pvalue = permutation_test_score(self.regressor, X, y, cv=cv, n_jobs=n_jobs, random_state=random_state)

        R_mean = mean(np.sqrt(R_SQUARED))
        R_std = std(np.sqrt(R_SQUARED))
        MAE_mean = mean(MAE)
        MAE_std = std(MAE)
        RMSE_mean = mean(RMSE)
        RMSE_std = std(RMSE)

        self._scores = dict([
                        ('R_mean'   , R_mean), 
                        ('R_std'    , R_std), 
                        ('MAE_mean' , MAE_mean), 
                        ('MAE_std'  , MAE_std),
                        ('RMSE_mean', RMSE_mean),
                        ('RMSE_std' , RMSE_std),
                        ('pvalue'   , pvalue)
                        ])

        return self.regressor.fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self.regressor.predict(X, *args, **kwargs)


  # source : https://rowannicholls.github.io/python/statistics/agreement/concordance_correlation_coefficient.html
def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator / denominator