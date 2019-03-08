# BCI Emotion Detection
Note: Will need EEGLAB MATLAB to run file

This repo presents code for detection of emotions/participants using EEG data collected through movie clips. An unsupervised Kmedoids Spil-and-Merge clustering algorithm was employed. The data is collected by IIT Gandhinagar Cognitive Science department using high density EEG equipment for 20 subjects for 9 emotional movie clips. This is followed by dimensionality reduction after which the data is divided into frequency bands - theta, alpha and beta. Following this, we applied our very own novel recursive clustering strategy similar to Hierarchical Kmeans algorithm. We studied the groups formed to realize the exceeding similarities in subject-wise emotions.

Publication:
S S Nath, D Mukhopadhyay, K P Miyapuram, Emotive Stimuli-triggered Participant-based Clustering Using a Novel Split-and-Merge Algorithm, CoDS-COMAD YRS 2019.
