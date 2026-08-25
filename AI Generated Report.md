\## Technical Report: Automated Noise Acquisition System

\*\*Prepared by:\*\* NAVIGATOR

\*\*Institution:\*\* University of Florida



\### Background

The goal of this project is to develop an automated noise measurement system for quantum optics experiments. The system uses a Thorlabs PDB230C balanced photodetector and a Keysight DSA91304A oscilloscope connected over LAN to collect noise measurements from Channel 3, which outputs the hardware difference signal:

V\_3(t) = V\_1(t) - V\_2(t)



\### Original Method and Limitations

The original approach used the oscilloscope's built-in GUI to manually monitor Channel 3 and record FFT marker values at 10 MHz with 100 FFT averages. Several fundamental issues were identified:

\- The 100-acquisition FFT average on the scope was not synchronized with the V3 balance condition. There was no guarantee that the averaged noise measurement corresponded to moments of good detector balance.

\- The alignment condition was determined by visual inspection, introducing human bias and poor reproducibility.

\- A single marker value was recorded per condition, providing no statistical uncertainty.

\- Averaging was performed implicitly in dBm display units rather than in linear watts, which is physically incorrect for noise power averaging.



\### Proposed Automated System

A Python script was developed that connects to the DSA91304A over LAN using PyVISA and automates the full acquisition pipeline. The key design decisions are:



\*\*Synchronized ordered pairs\*\*

For every short analysis window, the script computes both the alignment metric and the noise measurement from exactly the same Channel 3 samples, guaranteeing they correspond to the same physical moment. Every acquisition produces one ordered pair:

(|mean(V3)|, P\_noise(10 MHz))



\*\*Subwindow segmentation\*\*

Each long acquisition record is divided into non-overlapping 15 µs subwindows. This provides better time resolution for tracking mechanical alignment drift and increases the number of data points per acquisition. Each subwindow contains enough samples to resolve a 100 kHz noise bandwidth at 10 MHz.



\*\*Alignment metric\*\*

The alignment condition uses the arithmetic mean of V3: 

A\_align = |mean(V3)| <= x

This was specifically chosen over total RMS because RMS includes the noise being measured, which would cause the measured noise to be underestimated through selection bias.



\*\*FFT computed in Python\*\*

The power spectral density is calculated using `scipy.signal.periodogram` with a Hann window and DC detrending. This gives explicit control over spectral normalization. The voltage PSD is converted to power PSD, and the bin noise power is extracted for the 100 kHz effective noise bandwidth (ENBW).



\*\*Linear power averaging\*\*

All noise values are stored in watts. Averaging is performed in linear units before converting to dBm, as required for physically correct noise power analysis.



\### Data Output

Every qualifying ordered pair is written to a CSV file in real time. The complete FFT spectrum for each qualifying acquisition is saved as a compressed file for later reanalysis. A separate convergence script monitors the running mean noise power and stops acquisition automatically when the mean has stabilized within a user-defined tolerance.



\### Current Status

The acquisition script is ready for initial validation on the instrument. Verification of the physical output values against known measurements from the previous GUI-based approach is the immediate next step.

