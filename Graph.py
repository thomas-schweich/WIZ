import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.optimize import curve_fit
from GraphWindow import GraphWindow
from numbers import Number
from MathExpression import MathExpression


class Graph:
    __author__ = "Thomas Schweich"

    def __init__(self, window=None, title="", xLabel="", yLabel="", rawXData=np.array([0]), rawYData=np.array([0]),
                 xMagnitude=0, yMagnitude=0, autoScaleMagnitude=False, subplot=None):
        """Creates a Graph of specified data including a wide variety of methods for manipulating the data.

        To plot multiple graphs on the same axis, specify the same subplot. A subplot may optionally be specified
        when displaying a graph. Without one matplotlib.pyplot.plot() is used directly when plotting.
        Creates a point at (0, 0) by default.
        """
        self.window = window
        self.title = title
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.rawXData = rawXData
        self.rawYData = rawYData
        self.xMagnitude = xMagnitude
        self.yMagnitude = yMagnitude
        self.autoScaleMagnitude = autoScaleMagnitude
        self.subplot = subplot
        self.show = True
        # self.graphWindow = GraphWindow(self)
        self.mode = ""
        self.master = False
        # TODO Make .title vs. getTitle() consistent
        # TODO xData and yData functions

    '''
    def getMetaData(self):
        """Returns a dict of all data about the graph other than the raw data it uses"""
        return {
            "window": self.window,
            "title": self.title,
            "xLabel": self.xLabel,
            "yLabel": self.yLabel,
            "xMagnitude": self.xMagnitude,
            "yMagnitude": self.yMagnitude,
            "autoScaleMagnitude": self.autoScaleMagnitude,
            "subplot": self.subplot,
            "show": self.show,
            "graphWindow": self.graphWindow,
            "mode": self.mode
        }
    '''

    def getMetaData(self):
        """Returns a dict of all class data which is not a function and not a numpy array"""
        return {key: value for key, value in self.__dict__.items() if not key.startswith("__") and
                not callable(key) and not (key == "rawXData" or key == "rawYData" or key == "graphWindow" or
                                           key == "window" or key == "subplot")}

    def useMetaFrom(self, other):
        """Sets the metadata of this graph to the metadata of other"""
        self.__dict__.update(other.getMetaData())  # TODO Use in all factory functions

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
        """Sets the subplot that should plot this graph by default using plot()"""
        self.subplot = sbplt

    def show(self):
        """Sets .show to True"""
        self.show = True

    def hide(self):
        """Sets .show to False"""
        self.show = False

    def isShown(self):
        """Returns whether or not this Graph should be displayed"""
        return self.show

    def getTitle(self):
        return str(self.title)

    def getXLabel(self):
        return str(self.xLabel)

    def getYLabel(self):
        return str(self.yLabel)

    def setGraphMode(self, mode):
        """Sets the graphing mode

        Possible options are 'logy', 'logx', 'loglog', and 'scatter'
        """
        self.mode = mode

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

    def plot(self, subplot=None, mode=None, maxPoints=None):
        """Plots a PyPlot of the graph"""
        xMag, yMag = self.getMagnitudes()
        numPts = len(self.getRawData()[0])
        if maxPoints and numPts > maxPoints:
            step = int(numPts / maxPoints)
            print "Using step size: %d" % step
            xVals = self.getScaledMagData()[0][::step]
            yVals = self.getScaledMagData()[1][::step]
            print "Points plotted: %d" % len(xVals)
        else:
            xVals, yVals = self.getScaledMagData()
        if not mode: mode = self.mode
        if subplot:
            sub = subplot
        else:
            sub = self.subplot
        if mode == "scatter":
            (plt if not sub else sub).scatter(xVals, yVals)
        elif mode == "logy":
            (plt if not sub else sub).semilogy(xVals, yVals)
        elif mode == "logx":
            (plt if not sub else sub).semilogx(xVals, yVals)
        elif mode == "loglog":
            (plt if not sub else sub).loglog(xVals, yVals)
        else:
            (plt if not sub else sub).plot(xVals, yVals)  # , ",")
        if not sub:
            plt.xlabel((str(self.getXLabel()) + "x10^" + str(xMag) if xMag != 0 else str(self.getXLabel())))
            plt.ylabel((str(self.getYLabel()) + "x10^" + str(yMag) if yMag != 0 else str(self.getYLabel())))
            plt.title(str(self.getTitle()))
        else:
            sub.set_xlabel((str(self.getXLabel()) + "x10^" + str(xMag) if xMag != 0 else str(self.getXLabel())))
            sub.set_ylabel((str(self.getYLabel()) + "x10^" + str(yMag) if yMag != 0 else str(self.getYLabel())))
            sub.set_title(str(self.getTitle()))

    def scatter(self, subplot=None):
        """Shortcut for mode="scatter" default in plot()"""
        self.plot(subplot=subplot, mode="scatter")

    def getCurveFit(self, fitFunction):
        """Returns a Graph of fitFunction with fitted parameters"""
        forcedXMag, forcedYMag = self.getMagnitudes(forceAutoScale=True)
        setXMag, setYMag = self.getMagnitudes()
        xVals, yVals = self.getScaledMagData(forceAutoScale=True)
        fitParams, fitCoVariances = curve_fit(fitFunction, xVals, yVals, check_finite=False)  # , maxfev=100000)
        magAdjustment = forcedYMag - setYMag
        return Graph(self.window, rawXData=np.array(self.getRawData()[0]), rawYData=np.array(
            fitFunction(self.getScaledMagData(forceAutoScale=True)[0], *fitParams)) * 10 ** (magAdjustment + setYMag),
                     autoScaleMagnitude=self.autoScaleMagnitude, title="Fit for " + self.title, xLabel=self.xLabel,
                     yLabel=self.yLabel)
        # (raw vs. scaled - consult)

    def getFFT(self):
        """Returns a Graph of the Single-Sided Amplitude Spectrum of y(t)"""
        x, y = self.getRawData()
        sampleTime = x[1] - x[0]
        n = len(y)  # length of the signal
        k = np.arange(n)
        T = n * sampleTime
        frq = k / T  # two sides frequency range
        frq = frq[range(n / 2)]  # one side frequency range
        Y = fft(y) / n  # fft computing and normalization
        Y = Y[range(n / 2)]
        result = Graph(self.window, rawXData=frq, rawYData=abs(Y), title="FFT", xLabel="Freq (Hz)", yLabel="|Y(freq)|")
        result.setGraphMode("loglog")
        return result
        '''
        # Other attempt at FFT:
        x, y = self.getRawData()
        interval = x[1] - x[0]
        sampleFrq = 1.0 / interval
        totalTime = x[-1] - x[0]
        frqRes = 1 / totalTime
        n = len(y)
        frqVector = [i * frqRes for i in range(n / 2)]
        yFFT = fft(y)
        '''

    def convertUnits(self, xMultiplier=1, yMultiplier=1, xLabel=None, yLabel=None):
        """Returns a Graph with data multiplied by specified multipliers. Allows setting new labels for units."""
        return Graph(self.window, title=str(self.title) + " (converted)",
                     xLabel=(self.xLabel if not xLabel else xLabel),
                     yLabel=(self.yLabel if not yLabel else yLabel),
                     rawXData=self.getRawData()[0] * xMultiplier, rawYData=self.getRawData()[1] * yMultiplier,
                     autoScaleMagnitude=self.autoScaleMagnitude)

    def slice(self, begin=0, end=None, step=1):
        """Returns a Graph of the current graph's data from begin to end in steps of step.

        Begin defaults to 0, end to len(data)-1, step to 1.
        """
        end = len(self.getRawData()[0]) - 1 if not end else end
        return Graph(self.window, title=str(self.title) + " from point " + str(int(begin)) + " to " + str(int(end)),
                     xLabel=self.xLabel, yLabel=self.yLabel, rawXData=self.getRawData()[0][begin:end:step],
                     rawYData=self.getRawData()[1][begin:end:step], autoScaleMagnitude=self.autoScaleMagnitude)

    def onClick(self, event):
        """Opens this Graph's GraphWindow if the event is within its axes and was a double click"""
        if event.inaxes is self.subplot and event.dblclick:
            self.openWindow()

    def openWindow(self):
        """Opens this Graph's GraphWindow"""
        self.graphWindow = GraphWindow(self)  # TODO Why does it need to generate a new window each time?
        self.graphWindow.open()

    def isSameX(self, other):
        return np.array_equal(self.getRawData()[0], other.getRawData()[0])

    @staticmethod
    def useYForCall(function, *args):
        newArgs = list(args[:])
        graph = None
        for index, arg in enumerate(newArgs):
            try:
                newArgs[index] = arg.getRawData()[1]
            except AttributeError:
                pass
            else:
                if graph:
                    if len(args[index].getRawData()[0]) > len(graph.getRawData()[0]):
                        graph = args[index]
                else:
                    graph = args[index]
        try:
            graph.setRawData((graph.getRawData()[0], function(*newArgs)))
            return graph
        except AttributeError as a:
            raise MathExpression.ParseFailure(str(graph), a)

    def __repr__(self):
        """Returns the Graph's title"""
        return str(self.title)

    def __sub__(self, other):
        """Subtracts the y data of two graphs and returns the resulting Graph.

        Returns NotImplemented if used on a non-graph,
         non-number object or the data sets do not have the same x values.
        """
        if isinstance(other, Graph) and np.array_equal(self.getRawData()[0], other.getRawData()[0]):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], self.getRawData()[1] - other.getRawData()[1]))
            g.setTitle(self.getTitle() + " - " + str(other.getTitle()))
            return g
        elif isinstance(other, Number):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], self.getRawData()[1] - other))
            g.setTitle(self.getTitle() + " - " + str(other))
            return g
        else:
            return NotImplemented

    def __add__(self, other):
        """Adds the y data of two graphs and returns the resulting Graph

        Returns NotImplemented if used on a non-graph,
         non-number object or the data sets do not have the same x values."""
        if isinstance(other, Graph) and np.array_equal(self.getRawData()[0], other.getRawData()[0]):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], self.getRawData()[1] + other.getRawData()[1]))
            g.setTitle(self.getTitle() + " + " + str(other.getTitle()))
            return g
        elif isinstance(other, Number):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], self.getRawData()[1] + other))
            g.setTitle(self.getTitle() + " + " + str(other))
            return g
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiplies the y data of two graphs and returns the resulting Graph

        Returns NotImplemented if used on a non-graph,
         non-number object or the data sets do not have the same x values."""
        if isinstance(other, Graph) and np.array_equal(self.getRawData()[0], other.getRawData()[0]):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], self.getRawData()[1] * other.getRawData()[1]))
            g.setTitle(self.getTitle() + " * " + str(other.getTitle()))
            return g
        elif isinstance(other, Number):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], self.getRawData()[1] * other))
            g.setTitle(self.getTitle() + " * " + str(other))
            return g
        else:
            return NotImplemented

    def __div__(self, other):
        """Divides the y data of two graphs and returns the resulting Graph

        Returns NotImplemented if used on a non-graph,
         non-number object or the data sets do not have the same x values."""
        if isinstance(other, Graph) and np.array_equal(self.getRawData()[0], other.getRawData()[0]):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], self.getRawData()[1] / other.getRawData()[1]))
            g.setTitle(self.getTitle() + " / " + str(other.getTitle()))
            return g
        elif isinstance(other, Number):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], self.getRawData()[1] / other))
            g.setTitle(self.getTitle() + " / " + str(other))
            return g
        else:
            return NotImplemented

    def __pow__(self, other, modulo=None):
        """Takes the y data of this Graph to the power of a number, or another graphs's y data, returning the result

        !! Modulo argument not implemented !!"""
        # TODO Modulo
        if isinstance(other, Graph) and np.array_equal(self.getRawData()[0], other.getRawData()[0]):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], np.power(self.getRawData()[1], other.getRawData()[1])))
            g.setTitle(self.getTitle() + " ^ " + str(other.getTitle()))
            return g
        elif isinstance(other, Number):
            g = Graph(self.window)
            g.__dict__.update(self.getMetaData())
            g.setRawData((self.getRawData()[0], np.square(
                self.getRawData()[1]) if other == 2 else np.power(self.getRawData()[1], other)))
            g.setTitle(self.getTitle() + " ^ " + str(other))
            return g
        else:
            return NotImplemented

    def __len__(self):
        """Returns the number of x data points in the graph"""
        return len(self.getRawData()[0])


def create(xData, yData):
    return Graph(rawXData=xData, rawYData=yData)


def x(graph, index=None):
    if index:
        return graph.getRawData()[0][index]
    else:
        return graph.getRawData()[0]


def y(graph, index=None):
    if index:
        return graph.getRawData()[1][index]
    else:
        return graph.getRawData()[1]


def length(graph):
    return len(graph)


def getFFT(graph):
    gr = graph.getFFT() / ((2 * np.pi) ** .5)
    gr.setTitle("FFT (LabView Scale Factor)")
    return gr


def getDispFFT(graph):
    return ((getFFT(graph) ** 2) / (1.0 / x(graph, 1) - x(graph, 0) * length(graph))) ** .5



