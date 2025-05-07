
# basic packages
import yaml
import os

""" ===========================================================================

                    Public function 

============================================================================"""

def load_config(config_path=None):
    """
    Load the configuration file to set all the parameters 

    Parameters
    ----------
    config_path : string, optional
        Path to the configuration file.
        if no valid configuration file is given, the parameters are set to the
        default values.

    Returns
    -------
    PARAMS : dictionary
        Dictionary with all the parameters that are required 
    """    
    global CONFIG


    if os.path.isfile(str(config_path)): 
        try:
            with open(config_path, 'r') as f:
                CONFIG = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found. \n Load the default configuration.")
    else:
            print(f"Error: Configuration file '{config_path}' not found. \n Load the default configuration.")

    return CONFIG

def get_config() :
    return CONFIG
