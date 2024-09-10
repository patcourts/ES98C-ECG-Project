# ES98C-ECG-Project

- This repository exists to hold the code made whilst performing analysis necessary for my MSc thesis. 
- The aim of this project is to perform analysis on ECG signals within the PTB Diagnostic ECG database [1] to investigate the potential of automatic ECG classification into healthy and unhealthy groups.
- Two main methods were investigated in the report:
  - The SVM, which used paramterised signals as inputs,
  - The 1D CNN, which used the complete signals as inputs.
- The code within this repository pertains to the filtering of the PTB database, the denoising of the ECG signals and then the subsequent implementation of these methods.
- Jupyter notebooks are included, within the 'examples' folder as a rough preliminary investigation of methods whereby the main codebase is with python files within the 'src' folder.
- Also included within 'misc' folder in this repository is:
  * A poster designed for the BioMedEng24 conference at Queen Mary University of London summarise parts of the thesis
  * The completed report as written for the MSc solo research project module.

[1] https://physionet.org/content/ptbdb/1.0.0/
