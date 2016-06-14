import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math

class Graph:

    def __init__(self, title="", xLabel="", yLabel="", rawXData=np.array([0]), rawYData=np.array([0]), xMagnitude=0,
                 yMagnitude=0, autoScaleMagnitude=False):
        """Creates a graph including code for proper scaling into scientific notation and curve fitting.

        To plot multiple graphs on the same axis, simply refrain from subplotting.
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

    def setRawData(self, data):
        """Uses a tuple of (x data, y data) as the unscaled data of the graph."""
        self.rawXData, self.rawYData = data

    def getRawData(self):
        """Returns a tuple of (raw x data, raw y data)"""
        return self.rawXData, self.rawYData

    def setTitle(self, title):
        self.title = title

    def setXLabel(self, label):
        self.xLabel = label

    def setYLabel(self, label):
        self.yLabel = label

    def getMagnitudes(self, forceAutoScale=False):
        """Returns a tuple of the magnitudes of the first point of (x data, y data)

        forceAutoScale can be set to true to ensure that you receive the actual magnitudes of the first data point
        (for use when fitting curves and magnitudes as near as possible to 1 are needed)
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
            xMag = self.getMagnitudes()[0] if not forceAutoScale else self.getMagnitudes(forceAutoScale=True)[0]
        if not yMag:
            yMag = self.getMagnitudes()[1] if not forceAutoScale else self.getMagnitudes(forceAutoScale=True)[1]
        xData, yData = self.getRawData()
        return xData/10**xMag, yData/10**yMag

    def plot(self, subplot=None, scatter=False):
        """Plots a PyPlot of the graph"""
        xMag, yMag = self.getMagnitudes()
        xVals, yVals = self.getScaledMagData()
        if scatter:
            (plt if not subplot else subplot).scatter(xVals, yVals)
        else:
            (plt if not subplot else subplot).plot(xVals, yVals)
        plt.xlabel((str(self.xLabel) + "x10^" + str(xMag) if xMag != 0 else str(self.xLabel)))
        plt.ylabel((str(self.yLabel) + "x10^" + str(yMag) if yMag != 0 else str(self.yLabel)))
        plt.title(str(self.title))

    def scatter(self, subplot=None):
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
                     yLabel=self.yLabel)  # TODO TEST MORE
        # (raw vs. scaled - consult)

    def __sub__(self, other):
        if isinstance(other, Graph):
            return Graph(title=str(self.title) + " - " + str(other.title), xLabel=self.xLabel, yLabel=self.yLabel,
                         rawXData=self.rawXData, rawYData=self.getRawData[1]-other.getRawData()[1],
                         autoScaleMagnitude=self.autoScaleMagnitude)
        else:
            return NotImplemented
