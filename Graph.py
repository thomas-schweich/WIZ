import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from GraphWindow import GraphWindow

__author__ = "Thomas Schweich"


class Graph:
    def __init__(self, title="", xLabel="", yLabel="", rawXData=np.array([0]), rawYData=np.array([0]), xMagnitude=0,
                 yMagnitude=0, autoScaleMagnitude=False, subplot=None, root=None):
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
        self.root = root
        self.show = True
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

    def hide(self):
        self.show = False

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
        return xData / 10 ** xMag, yData / 10 ** yMag

    def plot(self, subplot=None, scatter=False):
        """Plots a PyPlot of the graph"""
        xMag, yMag = self.getMagnitudes()
        xVals, yVals = self.getScaledMagData()
        sub = (self.subplot if not subplot else subplot)
        if scatter:
            (plt if not sub else sub).scatter(xVals, yVals)
        else:
            (plt if not sub else sub).plot(xVals, yVals)
        if not sub:
            plt.xlabel((str(self.xLabel) + "x10^" + str(xMag) if xMag != 0 else str(self.xLabel)))
            plt.ylabel((str(self.yLabel) + "x10^" + str(yMag) if yMag != 0 else str(self.yLabel)))
            plt.title(str(self.title))
        else:
            sub.set_xlabel((str(self.xLabel) + "x10^" + str(xMag) if xMag != 0 else str(self.xLabel)))
            sub.set_ylabel((str(self.yLabel) + "x10^" + str(yMag) if yMag != 0 else str(self.yLabel)))
            sub.set_title(str(self.title))

    def scatter(self, subplot=None):
        """Shortcut for scatter=True default in plot()"""
        self.plot(subplot=subplot, scatter=True)

    def getCurveFit(self, fitFunction):
        """Returns a Graph of fitFunction with fitted parameters"""
        forcedXMag, forcedYMag = self.getMagnitudes(forceAutoScale=True)
        setXMag, setYMag = self.getMagnitudes()
        xVals, yVals = self.getScaledMagData(forceAutoScale=True)
        fitParams, fitCoVariances = curve_fit(fitFunction, xVals, yVals)  # , maxfev=100000)
        # print fitParams
        magAdjustment = forcedYMag - setYMag
        return Graph(rawXData=np.array(self.getRawData()[0]), rawYData=np.array(
            fitFunction(self.getScaledMagData(forceAutoScale=True)[0], *fitParams)) * 10 ** (magAdjustment + setYMag),
                     autoScaleMagnitude=self.autoScaleMagnitude, title="Fit for " + self.title, xLabel=self.xLabel,
                     yLabel=self.yLabel)
        # (raw vs. scaled - consult)

    def convertUnits(self, xMultiplier=1, yMultiplier=1, xLabel=None, yLabel=None):
        """Returns a Graph with data multiplied by specified multipliers. Allows setting new labels for units."""
        return Graph(title=str(self.title) + " (converted)", xLabel=(self.xLabel if not xLabel else xLabel),
                     yLabel=(self.yLabel if not yLabel else yLabel),
                     rawXData=self.getRawData()[0] * xMultiplier, rawYData=self.getRawData()[1] * yMultiplier,
                     autoScaleMagnitude=self.autoScaleMagnitude)

    def slice(self, begin=0, end=None, step=1):
        """Returns a Graph of the current graph's data from begin to end in steps of step.

        Begin defaults to 0, end to len(data)-1, step to 1.
        """
        end = len(self.getRawData()[0] - 1) if not end else end
        return Graph(title=str(self.title) + " from point " + str(int(begin)) + " to " + str(int(end)),
                     xLabel=self.xLabel, yLabel=self.yLabel, rawXData=self.getRawData()[0][begin:end:step],
                     rawYData=self.getRawData()[1][begin:end:step], autoScaleMagnitude=self.autoScaleMagnitude)

    def onClick(self, event):
        """Opens this Graph's GraphWindow if the event is within its axes and was a double click"""
        if event.inaxes is self.subplot and event.dblclick:
            print str(self.title) + " was clicked."
            self.openWindow()

    def openWindow(self):
        self.graphWindow.open()

    def __repr__(self):
        """Returns the Graph's title"""
        return str(self.title)

    def __sub__(self, other):
        """Subtracts the y data of two graphs and returns the resulting Graph.

        Returns a NotImplemented singleton if used on a non-graph object or the data sets are not of the same length.
        """
        if isinstance(other, Graph) and len(self.getRawData()) == len(other.getRawData()):
            return Graph(title=str(self.title) + " - " + str(other.title), xLabel=self.xLabel, yLabel=self.yLabel,
                         rawXData=self.rawXData, rawYData=self.getRawData()[1] - other.getRawData()[1],
                         autoScaleMagnitude=self.autoScaleMagnitude)
        else:

            return NotImplemented
