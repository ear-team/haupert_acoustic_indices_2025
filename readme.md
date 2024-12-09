# Is monitoring biodiversity with acoustic indices a cul-de-sac?

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Citation Badge](https://api.juleskreuer.eu/citation-badge.php?doi=xxx)](https://juleskreuer.eu/projekte/citation-badge/)

This repository contains supporting code for manuscript article xxx

The audio dataset used in the paper is archived on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14317014.svg)](https://doi.org/10.5281/zenodo.14317014) (NOT PUBLISHED YET)

If the code, even partially, is used for other purpose please cite the article 
`Haupert, S., Ducrettet, M., Sèbe, F., & Sueur, J. (2025). Is monitoring biodiversity with acoustic indices a cul-de-sac?. Journal xxx.`

## Setup and usage

All code was developed using an conda environment based on Python v3.10, but other versions may also work.

Download the `.zip` from Github (click on `code` then `Download Zip`) and extract all folders without changing the name of the folders neither rearrange the folder and sub-folders.

Then, download the audio dataset and the annotation file from Zenodo https://doi.org/10.5281/zenodo.14317014. Extract the `.zip` files in the directory `data`

Additional libraries required:
* tqdm
* scikit-maad
* scikit-learn
* seaborn

