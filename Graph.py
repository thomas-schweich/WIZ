import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure


__author__ = "Thomas Schweich"


class Graph:

    def __init__(self, title="", xLabel="", yLabel="", rawXData=np.array([0]), rawYData=np.array([0]), xMagnitude=0,
                 yMagnitude=0, autoScaleMagnitude=False, subplot=None):
        """Creates a Graph of specified data including a wide variety of methods for manipulating the data.

        To plot multiple graphs on the same axis, simply refrain from subplotting. A subplot may optionally be specified
        when displaying a graph.
        Creates a point at (0, 0) by default.
        """
        self.title = title
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.rawXData = rawXData
        self.rawYData = rawYData
        self.xMagnitude = xMagnitude
        self.yMagnitude = yMagnitude
        self.autoScaleMagnitude = autoScaleMagnitude
        self.subplot = subplot
        self.graphWindow = GraphWindow(self)


    def setRawData(self, data):
        """Uses a tuple of (x data, y data) as the unscaled data of the graph."""
        self.rawXData, self.rawYData = data

    def getRawData(self):
        """Returns a tuple of (raw x data, raw y data)"""
        return self.rawXData, self.rawYData

    def setTitle(self, title):
        """Sets the title of the graph"""
        self.title = title

    def setXLabel(self, label):
        """Sets the x label of the graph"""
        self.xLabel = label

    def setYLabel(self, label):
        """Sets the y label of the graph"""
        self.yLabel = label

    def setSubplot(self, sbplt):
        self.subplot = sbplt

    def getMagnitudes(self, forceAutoScale=False):
        """Returns the order of 10 magnitude of the data if autoScaleData is set to true

        Otherwise, it returns the specified scale (default 1)
        ForceAutoScale calculates the actual order of magnitude of the data no matter what.
        """
        if self.autoScaleMagnitude or forceAutoScale:
            rawX, rawY = self.getRawData()
            return (np.floor(np.log10(np.abs(rawX[0])))), (np.floor(np.log10(np.abs(rawY[0]))))
        else:
            return self.xMagnitude, self.yMagnitude

    def getScaledMagData(self, xMag=None, yMag=None, forceAutoScale=False):
        """Returns a tuple of (x data, y data) scaled according to x magnitude and y magnitude

        Uses object's set magnitudes by default.
        Meant to return a value between 1 and 10 for scientific notation.
        """
        if not xMag:
            xMag = (self.getMagnitudes(forceAutoScale=True)[0] if forceAutoScale else self.getMagnitudes()[0])
        if not yMag:
            yMag = (self.getMagnitudes(forceAutoScale=True)[1] if forceAutoScale else self.getMagnitudes()[1])
        xData, yData = self.getRawData()
        return xData/10**xMag, yData/10**yMag

    def plot(self, subplot=None, scatter=False):
        """Plots a PyPlot of the graph"""
        xMag, yMag = self.getMagnitudes()
        xVals, yVals = self.getScaledMagData()
        subplot = (self.subplot if not subplot else subplot)
        if scatter:
            (plt if not subplot else subplot).scatter(xVals, yVals)
        else:
            (plt if not subplot else subplot).plot(xVals, yVals)
        if not subplot:
            plt.xlabel((str(self.xLabel) + "x10^" + str(xMag) if xMag != 0 else str(self.xLabel)))
            plt.ylabel((str(self.yLabel) + "x10^" + str(yMag) if yMag != 0 else str(self.yLabel)))
            plt.title(str(self.title))
        else:
            subplot.set_xlabel((str(self.xLabel) + "x10^" + str(xMag) if xMag != 0 else str(self.xLabel)))
            subplot.set_ylabel((str(self.yLabel) + "x10^" + str(yMag) if yMag != 0 else str(self.yLabel)))
            subplot.set_title(str(self.title))

    def scatter(self, subplot=None):
        """Shortcut for scatter=True default in plot()"""
        self.plot(subplot=subplot, scatter=True)

    def getCurveFit(self, fitFunction):
        """Returns a Graph of fitFunction with fitted parameters"""
        forcedXMag, forcedYMag = self.getMagnitudes(forceAutoScale=True)
        setXMag, setYMag = self.getMagnitudes()
        xVals, yVals = self.getScaledMagData(forceAutoScale=True)
        fitParams, fitCoVariances = curve_fit(fitFunction, xVals, yVals)  # , maxfev=100000)
        print fitParams
        magAdjustment = forcedYMag-setYMag
        return Graph(rawXData=np.array(self.getRawData()[0]), rawYData=np.array(
            fitFunction(self.getScaledMagData(forceAutoScale=True)[0], *fitParams)) * 10 ** (magAdjustment + setYMag),
                     autoScaleMagnitude=self.autoScaleMagnitude, title=self.title, xLabel=self.xLabel,
                     yLabel=self.yLabel)
        # (raw vs. scaled - consult)

    def convertUnits(self, xMultiplier=1, yMultiplier=1, xLabel=None, yLabel=None):
        """Returns a Graph with data multiplied by specified multipliers. Allows setting new labels for units."""
        return Graph(title=str(self.title) + " (converted)", xLabel=(self.xLabel if not xLabel else xLabel),
                     yLabel=(self.yLabel if not yLabel else yLabel),
                     rawXData=self.getRawData()[0]*xMultiplier, rawYData=self.getRawData()[1]*yMultiplier,
                     autoScaleMagnitude=self.autoScaleMagnitude)

    def slice(self, begin=0, end=None, step=1):
        """Returns a Graph of the current graph's data from begin to end in steps of step.

        Begin defaults to 0, end to len(data)-1, step to 1.
        """
        end = len(self.getRawData()[0] - 1) if not end else end
        return Graph(title=str(self.title) + " from point " + str(begin) + " to " + str(end),
                     xLabel=self.xLabel, yLabel=self.yLabel, rawXData=self.getRawData()[0][begin:end:step],
                     rawYData=self.getRawData()[1][begin:end:step], autoScaleMagnitude=self.autoScaleMagnitude)

    def onClick(self, event):
        if event.inaxes is self.subplot:
            print str(self.title) + " was clicked."
            self.graphWindow.open()

    def __sub__(self, other):
        """Subtracts the y data of two graphs and returns the resulting Graph.

        Returns a NotImplemented singleton if used on a non-graph object or the data sets are not of the same length.
        """
        if isinstance(other, Graph) and len(self.getRawData()) == len(other.getRawData()):
            return Graph(title=str(self.title) + " - " + str(other.title), xLabel=self.xLabel, yLabel=self.yLabel,
                         rawXData=self.rawXData, rawYData=self.getRawData()[1]-other.getRawData()[1],
                         autoScaleMagnitude=self.autoScaleMagnitude)
        else:
            return NotImplemented


class GraphWindow(Tk.Frame):

    def __init__(self, graph, *args, **kwargs):
        Tk.Frame.__init__(self, *args, **kwargs)
        self.graph = graph
        self.window = None
        self.fitOptions = False
        self.fitButton = None
        self.fitQuadratic = None
        self.graph.plot()
        '''
        f = Figure(figsize=(5, 4), dpi=150)
        subPlot = f.add_subplot(111)
        self.graph.plot(subplot=subPlot)
        '''

    def open(self):
        """Opens a graph window only if there isn't already one open for this GraphWindow

        Thus only one window per Graph can be open using this method (assuming Graphs only have one GraphWindow)"""
        if self.window is None:
            self.window = Tk.Toplevel(self)
            self.window.wm_title(str(self.graph.title))
            label = Tk.Label(self.window, text="This is " + str(self.graph.title))
            label.pack(side="top", fill="both", expand=True, padx=100, pady=100)
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            self.populate()

    def close(self):
        """Destroys the window, sets the GraphWindows's window instance to None"""
        self.window.destroy()
        self.window = None

    def populate(self):
        self.fitButton = Tk.Checkbutton(self.window, command=self.toggleFitOptions)
        self.fitButton.pack()

    def toggleFitOptions(self):
        if self.fitOptions:
            self.fitOptions = False
            self.fitQuadratic.destroy()
        else:
            self.fitOptions = True
            self.fitQuadratic = self.addWidget(Tk.Button, command=self.fit)

    def addWidget(self, widgetType, **kwargs):
        wid = widgetType(self.window, **kwargs)
        wid.pack()
        return wid

    def fit(self):
        print "Fit called"


