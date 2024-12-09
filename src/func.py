from skimage.morphology import closing
import numpy as np
import matplotlib.pyplot as plt
from maad import sound, util, rois

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
        bin_l=BIN_L
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
    nROI = len(df_rois) / (tn[-1] / 60) 
    # ROIs coverage in %
    aROI = im_mask_filtered.sum() / (im_mask_filtered.shape[0]*im_mask_filtered.shape[1]) *100 

    if verbose :
        print('===> nROI : {:.0f}#/min | aROI : {:.2f}%'.format(round(nROI), aROI))

        """**************************** Display process ******************************"""

    if DISPLAY : 
        fig, ax = plt.subplots(4,1,figsize=(6/10*tn[-1],8), sharex=True)
        util.plot_spectrogram(Sxx_power, extent=EXT, ax=ax[0], ylabel ='', xlabel='', colorbar=False)
        util.plot_spectrogram(Sxx_dB_noNoise, log_scale=False, vmax = np.percentile(Sxx_dB_noNoise,99.9), vmin = np.percentile(Sxx_dB_noNoise,0.1), ylabel ='', xlabel='', extent=EXT, ax=ax[1], colorbar=False)
        util.plot2d(im_mask, ylabel ='', xlabel='', extent=EXT, ax=ax[2], colorbar=False)
        util.plot_spectrogram(Sxx_power, ylabel ='', extent=EXT, ax=ax[3], colorbar=False)
        if len(df_rois) > 0 :
            util.overlay_rois(im_ref=Sxx_power, 
                        rois = df_rois,
                        ylabel ='', 
                        extent=EXT, 
                        ax=ax[3], 
                        colorbar=False,)
            
    return nROI, aROI