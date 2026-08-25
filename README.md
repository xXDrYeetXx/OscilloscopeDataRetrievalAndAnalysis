Disclaimer: lots of ai was used to make this stuff

\# Oscilloscope Data Retrieval and Analysis



This directory contains a suite of Python scripts designed to automate the collection, balancing, and visualization of noise measurements from a balanced photodetector using a Keysight DSA91304A oscilloscope.



\## Included Scripts



\* \*\*`get\_data.py`\*\*  

&#x20; \*\*Purpose:\*\* The main data acquisition script.  

&#x20; \*\*Description:\*\* Connects to the oscilloscope via LAN, retrieves long Channel 3 waveform records, and divides them into shorter subwindows. For each subwindow, it checks if the signals are aligned (using the arithmetic mean of V3) and calculates the noise power using an FFT. Qualifying aligned pairs and their spectra are saved to CSV and NPZ files.



\* \*\*`converge\_noise.py`\*\*  

&#x20; \*\*Purpose:\*\* Automated acquisition until statistical stability.  

&#x20; \*\*Description:\*\* Runs a similar data collection loop to `get\_data.py`, but continually calculates the running average of the noise power. It automatically stops the data collection once the mean noise value stabilizes and converges within a user-defined tolerance.



\* \*\*`balance\_pairs.py`\*\*  

&#x20; \*\*Purpose:\*\* Post-processing dataset correction.  

&#x20; \*\*Description:\*\* Checks the saved data to ensure the accepted measurements aren't biased toward positive or negative voltage imbalances. If the data leans to one side of 0V, it randomly drops pairs from the overrepresented side until the dataset is perfectly centered.



\* \*\*`boxplot\_noise.py`\*\*  

&#x20; \*\*Purpose:\*\* Data visualization and quality check.  

&#x20; \*\*Description:\*\* Reads the saved qualifying pairs and generates side-by-side box plots of the noise power (in dBm) and the V3 alignment levels. This allows you to visually confirm that the data is well-centered around 0V and evaluate the spread and outliers of your noise measurements.

