# WIZ Is Zippy

WIZ is a data analysis program designed with the purpose of making operations on mid-to-large sized data sets such as slicing, fitting, fourier-transforming, and plotting with Matplotlib simple and quick, with no programming experience required.

### How To Be a WIZard

#### Windows
* Extract the .zip file into a folder of your choice
* Scroll near the bottom of the folder
* Right click on WIZ.exe > Send to... > Desktop (create shortcut)
* Double-click the desktop icon

### Mac and Linux
* Binaries coming soon. Python code requires Scipy, NumPy, Pandas, Tkinter, and Matplotlib.

### Basic Usage
* To load raw data into the program, click "Load Raw Data" in the initial window which appears when starting the program
* Select a text file containing your raw data, with independant data in one column and dependant data in another
    * Most file formats such as .csv and tab-delimited data are supported by default
* The first ten lines of your data will be displayed
    * If the data displays improperly, try clicking load anyways. It is possible that the preview could be wrong, but the data will still be correctly interpereted.
    * Headers will likely not display correctly. All that matters is whether or not they exist, not what column they are in.
* If your data has headers, check "data contains headers". It will not load properly if you don't.
* Unless any infinite or non-numeric values matter to your data, or you plan to remove them yourself, it is reccommended that you leave "clean infs and NaNs" checked. The will remove any non-numeric values from your data, also deleting the value corresponding to them. (i.e. if you have a point which is (+inf, 73.0909), the entire coordinate pair will simply be removed, not just +inf, which would result in a shorter x-column than y-column).
* There is generally no reason not to read data in chunks, unless you've already tried it and it hasn't worked/has taken too long. Normally it should speed up the loading process, not slow it down, and conserve memory. The chunk size can be adjusted in programsettings.json['Chunk Size'] (default 100,000 points at a time loaded).


### Copyright Information

Copyright (c) 2016 Thomas Schweich

TODO: License, open source??

