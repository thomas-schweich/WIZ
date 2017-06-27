# WIZ Is Zippy

WIZ is a data analysis program designed with the purpose of making operations on mid-to-large sized data sets such as slicing, fitting, fourier-transforming, and plotting with Matplotlib simple and quick, with no programming experience required. Extensive documentation can be found [in the wiki][wiki].  

WIZ is in beta stages and is prone to bugs! However, it can still be a very useful tool. Just check and double-check anything that you do with WIZ, and note that some things may not be seamless.

## How To Be a WIZard
The latest releases of WIZ can be found [here][releases].

#### Windows
* Follow the instructions given by double-clicking [the latest installer.][releases]
* Anaconda is reccomended if you would prefer to run the code natively.

#### Mac and Linux
* Binaries coming soon. Python code requires SciPy, NumPy, Pandas, Tkinter, and Matplotlib.

### Basic Usage
* To load raw data into the program, click "Load Raw Data" in the initial window which appears when starting the program
* Select a text file containing your raw data, with independent data in one column and dependant data in another
    * Most file formats such as .csv and tab-delimited data are supported by default
* The first ten lines of your data will be displayed
    * If the data displays improperly, try clicking load anyways. It is possible that the preview could be wrong, but the data will still be correctly interpreted.
    * Headers will likely not display correctly. All that matters is whether or not they exist, not what column they are in.
* If your data has headers, check "data contains headers". Otherwise, the data will not load properly.
* Unless any infinite or non-numeric values matter to your data, or you plan to remove them yourself, it is recommended that you leave "clean infs and NaNs" checked. The will remove any non-numeric values from your data, also deleting the value corresponding to them. (i.e. if you have a point which is (+inf, 73.0909), the entire coordinate pair will simply be removed, not just +inf, which would result in a shorter x-column than y-column).
* There is generally no reason not to read data in chunks, unless you've already tried it and it hasn't worked/has taken too long. Normally it should speed up the loading process, not slow it down, and conserve memory. The chunk size can be adjusted in programSettings.json under "Chunk Size" (default 100,000 points at a time loaded).
* After clicking load, you will be prompted to select a slice of data to work with in your project. Leaving the default values, and clicking "create project" will plot the entire set of data in your file. If you wish to use only part of your data, you can either use "index" mode, in which you specify point numbers starting from zero as your starting and stopping values, or "x-value" mode, in which you specify the nearest x-values to your desired start and end points. **This setting only works for data with strictly ascending x-values.**
* After clicking "create project," a new window will appear containing a plot of your selected data. Double clicking the graph brings up the analysis interface for the data plotted.
    * All options displayed perform their respective operations on the original graph, shown to the _left_.
    * The results of each operation are shown on the _right_.
    * All operations are performed point for point. 
    * Double clicking the graph on the _right_ creates a window allowing you to re-title and relabel the graph
    * Once you have performed the desired operation on the graph, you may choose to plot your result either on the same axis as the one which you selected, a new axis, or to replace your old graph with the new one which you generated. Replacing can be useful if, for instance, you wish to simply change the title of your graph. These operations are performed using the buttons on the bottom left of the screen.
    * You can preserve a graph, but choose not to display it in your project by un-checking "show". You can permanently remove a graph from the project by clicking "Delete Graph".

#### User Written Expressions
Any operations which cannot be reached through the GUI of the analysis interface can be reached through a user written expression. User written expressions can be written in the "Custom Expression" section of a graph's analysis interface, as well as during the creation of a project template. They are written in a C-like style. To access the data of an existing graph from within a user written expression, use the drop-down menu below the text box. It will insert `<Name Of Graph>` where "`Name Of Graph`" is substituted for the title of the graph which you wish to reference. To evaluate an expression, click the "Parse" button. If your expression evaluates to a new graph, it will be plotted on the right. Otherwise, your result will be printed in string form in a popup window. User written expressions have access to the following pre-defined operations and functions, where `<Graph>` is assumed to be any generic graph:

* `^`, `/`, `*`, `+`, and `-` perform powers, division, multiplication, addition, and subtraction respectively. Order of operations is enforced as follows: `^` > `/` = `*` > `+` = `-`. Additionally, you may group operations using parenthesis as normal, so `(3 + 2) / 4` evaluates to `1.25`.
* `x(<Graph>[, point_index])` returns only the x-column of `<Graph>`'s data in the form of a NumPy array. If the optional second argument is provided, it returns the float-value at the index of `point_index`.
* `y(<Graph>[, point_index])` behaves in much the same way as the above function, except it uses the y-column of `<Graph>`'s data.
* `length(<Graph>)` returns an integer representing the number of coordinates in `<Graph>`. It can also be passed an array, in which case it returns the length of the array.
* `getSlice(<Graph>, start, stop[, step])` returns a graph of the section of coordinates in `<Graph>` between integer indices `start` and `stop`. If the optional fourth argument is provided, the result is a selection which "skips" by `step`, i.e. setting `step` to `2` would yield every _other_ point in the range, `3` would yield every _third_ point in the range, etc.
* `create(x_data, y_data)` returns a graph with `x_data` as its independent data set and `y_data` as its dependant data set, where `x_data` and `y_data` are each arrays. This is useful for creating graphable objects from the results of NumPy functions without the need for a "base" existing graph.
* `linearFit(<Graph>)`, `quadraticFit(<Graph>)`, `cubicFit(<Graph>)`, and `quarticFit(<Graph>)` each return a graph representing the best fit of `<Graph>` according to their respective order of polynomial.
* `getFFT(<Graph>)` returns a graph representing the single-sided amplitude spectrum of `<Graph>` who's units are scaled to be compatible with the default scaling of NI LabView.

Any names not recognized by the parser will be looked up in the namespace of [NumPy](http://www.numpy.org/), and failing that, the namespace of Python's [math](https://docs.python.org/2/library/math.html) library. So, for instance, the expression `sin(pi/2)` is equivelant to writing the following expression in python:
```
import numpy
numpy.sin(numpy.pi/2)
```
The expression would evaluate to `1`. NumPy's documentation can be found [here](http://docs.scipy.org/doc/numpy/reference/). The namespace lookup feature makes user written expressions in WIZ extremely powerful.

For more information, see [WIZ's wiki][wiki].

### Credits
WIZ's development was made possible through extensive usage of the following open source projects:

Project: [Python](http://www.python.org/)  
License: https://www.python.org/download/releases/2.7/license/

Project: [SciPy](https://www.scipy.org/)  
License: https://www.scipy.org/scipylib/license.html

Project: [Matplotlib](http://matplotlib.org/)  
License: http://matplotlib.org/users/license.html

Project: [Tcl/Tk](https://www.tcl.tk/)  
License: https://www.tcl.tk/software/tcltk/license.html

Project: [Pandas](http://pandas.pydata.org/)  
License: http://pandas.pydata.org/pandas-docs/stable/overview.html#license

Special thanks to Josh Grebler for the clever name and logo.

### Copyright Information

Copyright (c) 2016 Thomas Schweich

TODO: License, open source??

[releases]: https://github.com/thomas-schweich/WIZ/releases
[wiki]: https://github.com/thomas-schweich/WIZ/wiki

