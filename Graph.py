import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import ttk as Tk
import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from copy import copy

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


from MainWindow import MainWindow


class GraphWindow(Tk.Frame):
    def __init__(self, graph, *args, **kwargs):
        """A frame object who's open() method creates a Tk.Toplevel (new window) with its contents"""
        Tk.Frame.__init__(self, *args, **kwargs)
        self.widgets = {}
        self.graph = graph
        self.newGraph = None
        self.newSubPlot = None
        self.root = self.graph.root
        self.window = None
        self.radioVar = Tk.IntVar()
        self.fitBox = None
        self.sliceBox = None
        self.addBox = None
        self.multBox = None
        self.canvas = None
        self.f = None
        self.rbFrame = None
        self.optionsFrame = None
        self.pack()


    def open(self):
        """Opens a graph window only if there isn't already one open for this GraphWindow

        Thus only one window per Graph can be open using this method (assuming Graphs only have one GraphWindow)"""
        if self.window is None:
            self.window = Tk.Toplevel(self)
            self.window.wm_title(str(self.graph.title))
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            self.f = Figure(figsize=(2, 1), dpi=150)
            graphSubPlot = self.f.add_subplot(121)
            self.graph.plot(subplot=graphSubPlot)
            self.newSubPlot = self.f.add_subplot(122)
            self.newGraph = copy(self.graph)
            self.newGraph.setTitle("Transformation of " + str(self.graph.title))
            self.newGraph.plot(subplot=self.newSubPlot)
            self.canvas = FigureCanvasTkAgg(self.f, self.window)
            self.canvas.show()
            self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
            self.optionsFrame = Tk.Frame(self.window)
            self.optionsFrame.pack(side=Tk.BOTTOM, fill=Tk.BOTH)
            self.rbFrame = Tk.Frame(self.window)
            self.rbFrame.pack(side=Tk.BOTTOM, fill=Tk.BOTH)
            self.populate()
            self.refreshOptions()

    def close(self):
        """Destroys the window, sets the GraphWindows's Toplevel instance to None"""
        self.window.destroy()
        self.window = None

    def populate(self):
        """Adds all widgets to the window in their proper frames, with proper cascading"""
        # FIT
        self.fitBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions, text="Fit Options",
                                     variable=self.radioVar, value=0)
        self.fitBox.val = 0
        self.addWidget(Tk.Button, parent=self.fitBox, command=self.quadraticFit, text="Quadratic Fit")
        self.addWidget(Tk.Button, parent=self.fitBox, command=self.linearFit, text="Linear Fit")

        # SLICE
        self.sliceBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions, text="Slice Options",
                                       variable=self.radioVar, value=1)
        self.sliceBox.val = 1
        sliceVar = Tk.IntVar()
        byIndex = self.addWidget(Tk.Radiobutton, parent=self.sliceBox,
                                 text="By index (from 0 to " + str(len(self.graph.getRawData()[0])) +
                                      ")", variable=sliceVar, value=0)
        byXVal = self.addWidget(Tk.Radiobutton, parent=self.sliceBox,
                                text="By nearest x value (from " + str(self.graph.getRawData()[0][0]) + " to " + str(
                                    self.graph.getRawData()[0][len(self.graph.getRawData()[0]) - 1]) + ")",
                                variable=sliceVar, value=1)

        start = self.addWidget(Tk.Entry, parent=self.sliceBox)
        start.insert(0, "Start")
        end = self.addWidget(Tk.Entry, parent=self.sliceBox)
        end.insert(0, "End")
        self.addWidget(Tk.Button, parent=self.sliceBox, command=lambda: self.addSlice(sliceVar, start.get(), end.get()),
                       text="Preview")

        # ADD
        self.addBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions,
                                     text="Add/Subtract Graphs", variable=self.radioVar, value=2)
        self.addBox.val = 2

        # MULTIPLY
        self.multBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions,
                                      text="Multiply/Divide Graphs", variable=self.radioVar, value=3)
        self.multBox.val = 3

    def refreshOptions(self):
        """Refreshes the displayed options based on the currently selected Radiobutton"""
        for k, v in self.widgets.iteritems():
            for widget in v:
                if widget.winfo_exists():
                    widget.pack_forget()
            # Main Radiobuttons
            try:
                if k.val == self.radioVar.get():
                    for widget in v:
                        widget.pack(side=Tk.TOP)
            except AttributeError:
                pass

    def addWidget(self, widgetType, parent=None, *args, **kwargs):
        """Adds a widget to the window.

        If a parent is specified, it will be placed in the widgets widgets dict under its parent.
        If no parent is specified, it will get its own entry in widgets, and immediately be packed.
        If a parent is specified which doesn't already have its own entry, one will be created for it. Thus, cascading
        trees of widgets can be created. A widget may be both a parent and a child.
        All values are added in lists so that a parent widget may have more than one child.
        """
        if not parent:
            wid = widgetType(self.rbFrame, *args, **kwargs)
            self.widgets[wid] = []
            wid.pack(expand=True, side=Tk.LEFT)
        else:
            wid = widgetType(self.optionsFrame, *args, **kwargs)
            if self.widgets[parent] in self.widgets.values():
                self.widgets[parent].append(wid)
            else:
                self.widgets[parent] = [wid]
        print self.widgets[wid if not parent else parent]
        return wid

    def plotWithReference(self, graph):
        """Plots the graph while maintaining a copy of the original graph on the same axes"""
        self.f.delaxes(self.newSubPlot)
        self.newSubPlot = self.f.add_subplot(122)
        referenceGraph = copy(self.graph)
        self.newGraph = graph
        referenceGraph.plot(subplot=self.newSubPlot)
        self.newGraph.plot(subplot=self.newSubPlot)
        self.canvas.show()

    def plotAlone(self, graph):
        """Replaces the plot of the original graph in the new graph window with the graph specified"""
        self.f.delaxes(self.newSubPlot)
        self.newSubPlot = self.f.add_subplot(122)
        self.newGraph = graph
        self.newGraph.plot(subplot=self.newSubPlot)
        self.canvas.show()

    def quadraticFit(self):
        self.plotWithReference(self.graph.getCurveFit(fitFunction=lambda x, a, b, c: a * x ** 2 + b * x + c))

    def linearFit(self):
        self.plotWithReference(self.graph.getCurveFit(fitFunction=lambda x, a, b: a * x + b))

    def addSlice(self, tkVar, begin, end):
        if tkVar.get() == 0:
            self.plotAlone(self.graph.slice(begin=float(begin), end=float(end)))
        elif tkVar.get() == 1:
            results = np.searchsorted(self.graph.getRawData()[0], np.array([np.float64(begin), np.float64(end)]))
            self.plotAlone(self.graph.slice(begin=results[0], end=results[1]))
