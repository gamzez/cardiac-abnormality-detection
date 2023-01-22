## Cardiac Abnormality Detection
**A feature-extraction-based classification applied to heart sound records**: spectral-, 
cepstral- and time-domain features of phonocardiogram records are extracted and fed into 
relatively simpler classifiers. Feature selection is done such that only the features that show a 
separation among normal and abnormal heart signals are accumulated for classification.
Logistic regression and ensemble learning based classifiers (Random Forest and AdaBoost) 
are trained and tested.

### Supplementary Material Description:

This repo is the supplementary material to my master's thesis. It consists of two main parts:
 - dataset: pcg records are saved as .wav files. Additionally, .txt files are used within scripts for easy query of the records.
 - codes: the scripts which were used to generate the results are given in this folder. Scripts can be listed as:
	* utils.py and segmentation_utils.py: used within other scripts and provide necessary functions to be utilized.
	* roc_curve_of_bandpower_ratio.ipynb: used to extract bandpower ratio features and visualize roc curve based on single-feature classification. 
	* manual_correction_of_segmentation.ipynb and cardiac_cycle_segmentation.ipynb: used to extract time domain features of cardiac phases. 
	* classification scripts: three classification scripts have been provided. All of them do feature extraction and classification as well as K-Fold cross validation. The difference is one is without chunking, one is with overlapped chunks and the last one is with non-overlapping chunks. The reason why it is provided in three separate scripts is because the notebook becomes so long and hard to follow if all done at once. 

Before running the python notebooks, requirements given in the requirements.txt file has to be installed. 