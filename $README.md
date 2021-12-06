# Lab-3-Experimental-Physics-PHAS0051-HPGe-Detector-for-Nuclear-Spectroscopy
This lab involves tons of data analysis, and thus Python script is used extensively. 
This main purpose is for records.

Here will be two main files: Spectra.py and Exe.ipynb,
the first of which encodes the main tool, a massive python class,
and the second is an example Python notebook showing how to use the class defined in Spectra.py.
These two files will be updated from time to time.

The othor .txt files are lab data sets to be analyzed using class spectra. 
Data shown as a list of counts order according to the channel number, they are uncalibrated.

## Workflow of the class 
1. Calibration
`cali_exam` to view the uncalibrated spectra, identifying a range of data including a specific photopeak;
`cali_fit_peak` fits a Gaussian model to the ranged data and returns the fitted peak position with error;
`cali_add_peak` passing the peak found in `cali_fit_peak()` to the instance variable for further calibraion;
`get_cali` with more than 2 peaks(in ch's) recorded, given a set of reference peaks, the function will calibrate the parameters for energy scale by linear regression;
`get_E_scale` receiving calibration from the other source.
If calibrated, the methods in analysis part can be utilized.

3. Spectra Analysis
`spectra()` plots the calibrated spectra;
`fit_peak` does the same in calibration but with energy scale;
`add_peaks` to record photopeaks obtained;
`label_peaks` labels photopeaks found on an axes you choose;
`ComptonEdge` indicates the compto edges of peaks identified on an axes you choose;

the other functions will be updated to the file according to the progress in the lab.
Plan: `eff`,`E_resol`
