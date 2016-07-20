import matplotlib
matplotlib.use("TkAgg")
import sys
import os
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from Graph import Graph
from functools import partial
import tkFileDialog
import math
import json


class MainWindow(Tk.Tk):
    __author__ = "Thomas Schweich"

    def __init__(self, graphs=None, *args, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        Tk.Tk.__init__(self, *args, **kwargs)
        with open('programSettings.json', 'r') as settingsFile:
            self.settings = json.load(settingsFile)
        if not graphs: graphs = []
        plt.style.use(self.settings["Style"])
        self.wm_title("WIZ")
        self.defaultWidth, self.defaultHeight = self.winfo_screenwidth(), self.winfo_screenheight() * .9
        self.geometry("%dx%d+0+0" % (self.defaultWidth, self.defaultHeight))
        self.graphs = graphs
        self.buttons = []
        self.topFrame = Tk.Frame(self)
        self.topFrame.pack(side=Tk.TOP, fill=Tk.X)
        self.buttonFrame = Tk.Frame(self.topFrame)
        self.buttonFrame.pack(side=Tk.TOP, expand=1, fill=Tk.X)
        self.bottomFrame = Tk.Frame(self)
        self.bottomFrame.pack(side=Tk.BOTTOM)
        self.saveButton = Tk.Button(self.bottomFrame, text="Save", command=self.saveProject)
        self.saveButton.pack()
        matplotlib.rcParams["agg.path.chunksize"] = self.settings["Plot Chunk Size"]
        self.fig = Figure(figsize=(5, 4), dpi=self.settings['DPI'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", lambda event: self.onClick(event))
        # self.canvas.mpl_connect("key_press_event", lambda event: self.on_key_event(event))  # Buggy??

    def _quit(self):
        """Closes the MainWindow"""
        self.root.quit()
        self.root.destroy()

    def setGraphs(self, graphs):
        """Sets this window's list of graphs"""
        self.graphs = graphs

    def saveProject(self):
        """Saves this window's graphs and """
        path = tkFileDialog.asksaveasfilename(defaultextension=".npz",
                                              filetypes=[("Numpy", ".npz")])
        #with open(path, "a+") as p:
        rawdata = []
        metadata = []
        for i, axis in enumerate(self.graphs):
            rawdata.append([])
            metadata.append([])
            for j, graph in enumerate(axis):
                rawdata[i].append([])
                metadata[i].append([])
                rawdata[i][j] = graph.getRawData()
                metadata[i][j] = graph.getMetaData()
        print "Length of raw data %d" % len(rawdata), metadata
        np.savez(path, np.array(rawdata), np.array(metadata))

    @staticmethod
    def loadProject(path, destroyTk=None):
        """Loads the .npz file at path and creates a MainWindow with the data plotted

        The .npz file must be in the format
        npz["arr_0"] = 2d array of graph data grouped by axis
        and npz["arr_1"] = associated metadata in dict at corresponding index for each graph
        """
        b = np.load(path, mmap_mode="r+")
        rawData, metaData = b["arr_0"], b["arr_1"]
        if destroyTk:
            destroyTk.quit()
            destroyTk.destroy()
        graphs = []
        window = MainWindow()
        for i in range(len(rawData)):
            graphs.append([])
            for j in range(len(rawData[i])):
                gr = Graph()
                gr.setRawData(rawData[i][j])
                gr.window = window
                for att in metaData[i][j]:
                    setattr(gr, att, metaData[i][j][att])
                graphs[i].append(gr)
        window.setGraphs(graphs)
        window.plotGraphs()
        window.mainloop()

    def on_key_event(self, event):
        print('you pressed %s' % event.key)
        key_press_handler(event, self.canvas, self.toolbar)

    @staticmethod
    def loadData(path, clean=True, chunkRead=True, chunkSize=100000):
        """Loads data depending on file type, returning the resulting numpy array.

        With clean=True, removes non-finite values from the data stored at the path
        With chunkRead=True, the number of lines in the file are estimated and a memmap is created to store the data.
        The data is then loaded into the memmap 100,000 points at a time.
        """
        ftype = path[path.rfind("."):]
        if ftype == ".npy":
            xData, yData = np.load(path, mmap_mode="r+")
        else:
            if chunkRead:
                lineSize = 0
                with open(path) as f:
                    '''
                    for i, line in enumerate(f):
                        if i == 100000:
                            lineSize += sys.getsizeof(line)
                            break
                        else:
                            lineSize += sys.getsizeof(line)
                    '''
                    numLines = sum(1 for line in f)
                    f.seek(0, 0)
                '''
                if lineSize:
                    print "Line size: %d" % lineSize
                    approxLines = numLines  # os.path.getsize(path) / (lineSize / 100000)
                    print "approxLines: %d" % approxLines
                else:
                    raise ValueError("Couldn't find first line")
                '''
                # Open memmap 10% larger than the estimated size
                if not os.path.exists("/tmp"):
                    os.makedirs("/tmp")
                with open("/tmp/arr.npy", 'w+') as tempFile:
                    mmap = open_memmap(tempFile.name, mode='w+', dtype=np.float64, shape=(numLines, 2))
                # Parse 100000 points at a time to avoid overflow
                n = 0
                for chunk in pd.read_table(path, chunksize=chunkSize, dtype=np.float64, usecols=[0, 1], header=None):
                    mmap[n: n + chunk.shape[0]] = chunk.values
                    n += chunk.shape[0]
                    print "Chunk read"
                xData, yData = np.trim_zeros(mmap[:,0]), np.trim_zeros(mmap[:,1])
            else:
                xData, yData = np.loadtxt(path, unpack=True, dtype=np.float64)
        if clean:
            xFinite = np.isfinite(xData)
            yFinite = np.isfinite(yData)
            finitePoints = np.logical_and(xFinite, yFinite)
            xData = xData[finitePoints]
            yData = yData[finitePoints]
        return xData, yData
        # TODO .sac files, HDF5 format

    def addGraph(self, graph, parent=None, plot=True):
        """Adds a graph to this MainWindow's .graphs list, plotting it unless plot is set to false"""
        graphs = [gr for ax in self.graphs for gr in ax]
        n = sum(1 for gr in graphs if gr.getTitle() == graph.getTitle())
        if n:
            graph.setTitle("%s (%d)" % (str(graph.getTitle()), n))
        self.addGraphToAxisList(self.graphs, graph, parent=parent)
        if plot:
            self.plotGraphs()
        return graph

    @staticmethod
    def addGraphToAxisList(axisList, graph, parent=None):
        """Takes a list and adds a graph to it, adding it to its parent's sub-list if specified

         Each sub-list represents an axis. If a parent is specified but doesn't exist in the list, it is added as though
          no parent were specified."""
        if parent:
            for axis in axisList:
                if len(axis) > 0:
                    for g in axis:
                        if g is parent:
                            axis.append(graph)
                            return axisList
        axisList.append([graph])
        return axisList

    def removeGraph(self, graph, plot=True):
        """Removes the graph from the MainWindow's .graphs list, re-plotting unless plot is False"""
        for axis in self.graphs:
            for gr in axis:  # if graph in axis:
                if graph is gr:
                    axis.remove(graph)
            if len(axis) < 1:
                self.graphs.remove(axis)
        print "Graph list: %s" % str(self.graphs)
        if plot: self.plotGraphs()

    def onClick(self, event):
        """If event.dblclick, calls promptSelect() with the axis designated by event.inaxis"""
        print "Graphs list: %s" % str(self.graphs)
        if event.dblclick:
            for axis in self.graphs:
                for graph in axis:
                    if event.inaxes is graph.subplot:
                        self.promptSelect(axis)
                        return

    def promptSelect(self, graphsInAxis):
        """Creates a window prompting the user to select a graph from the axis if len(graphsInAxis) > 1

        otherwise opens the graph's GraphWindow"""
        graphsInAxis = [gr for gr in graphsInAxis if gr.isShown()]
        print "Graphs in axis: %s" % str(graphsInAxis)
        if len(graphsInAxis) > 1:
            window = Tk.Toplevel()
            Tk.Label(window, text="Available Graphs on this axis:").pack()
            for graph in graphsInAxis:
                Tk.Button(window, text=str(graph.title),
                          command=partial(self.openGrWinFromDialogue, graph, window)).pack()
        else:
            graphsInAxis[0].openWindow()

    @staticmethod
    def openGrWinFromDialogue(graph, window):
        """Opens graph's GraphWindow and destroys window"""
        graph.openWindow()
        window.destroy()

    def plotGraphs(self):
        """Plots all graphs in the MainWindows .graphs list, creating a button for each which isn't shown"""
        self.fig.clear()
        self.clearButtons()
        for axis in self.graphs:
            for graph in axis:
                if not graph.isShown():
                    self.buttons.append(Tk.Button(self.buttonFrame, text=str(graph.title), command=graph.openWindow))
        axesToShow = []
        for axis in self.graphs:
            if len(axis) > 0:
                for graph in axis:
                    if graph.isShown():
                        axesToShow.append(axis)
                        break
        length = len(axesToShow)
        rows = math.ceil(length / 2.0)
        subplots = [self.fig.add_subplot(rows, 1 if length == 1 else 2, index + 1)
                    for index in range(0, length)]
        for idx, axis in enumerate(axesToShow):
            for g in axis:
                if g.isShown():
                    g.setSubplot(subplots[idx])
                    g.plot()
        self.canvas.draw()
        for button in self.buttons:
            button.pack(side=Tk.LEFT, fill=Tk.X, expand=1)

    def clearButtons(self):
        """Destroys all buttons in .buttons"""
        for b in self.buttons:
            b.destroy()
        del self.buttons
        self.buttons = []

    @staticmethod
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def sinusoid(x, a, b, c, d):
        return a * (np.sin(b * x + c)) + d


def generateEQPGraphs(window):
    """Sample method for generating default graphs in a chain"""
    xVals, yVals = window.loadData("EQPtest_23Nov2015to26Nov2015UTC_part.txt")  # "Tyson.FI2.day280.TOR2.txt")
    unaltered = window.addGraph(
        Graph(window, title="Unaltered data", rawXData=xVals, rawYData=yVals,
              yLabel="Amplitude (px)", xLabel="Time (s)"), plot=False)
    fit = window.addGraph(unaltered.getCurveFit(window.quadratic), parent=unaltered, plot=False)
    driftRm = window.addGraph(unaltered - fit, plot=False)
    driftRm.setTitle("Drift Removed")
    unitConverted = window.addGraph(driftRm.convertUnits(yMultiplier=1.0 / 142857.0, yLabel="Position (rad)"),
                                    plot=False)
    slice = unitConverted
    sliceX, sliceY = slice.getRawData()
    xFourier = np.linspace(sliceX[0], sliceX[-1], num=len(sliceX))
    sliceAveraged = window.addGraph(Graph(window, title="Averaged Intervals", rawXData=xFourier, rawYData=sliceY))
    sliceFFT = sliceAveraged.getFFT()
    sliceFFT.setTitle("FFT")
    window.addGraph(sliceFFT)
    sliceFFTConverted = window.addGraph(sliceFFT / ((2 * math.pi) ** .5), parent=sliceFFT)
    fftY = sliceFFT.getRawData()[1]
    sampleFreq = 1 / (fftY[1] - fftY[0])
    N = len(fftY)
    # Marvel at this beautiful line of code:
    convertedFFT = ((sliceFFT / (2 * math.pi) ** 2) / (sampleFreq * N)) ** .5
    convertedFFT.setGraphMode("loglog")

    window.addGraph(convertedFFT)


def generateTestGraphs(window):
    """Sample method for generating default graphs in a chain"""
    xVals, yVals = window.loadData("EQPtest_23Nov2015to26Nov2015UTC_part.txt")  # "Tyson.FI2.day280.TOR2.txt")
    unaltered = window.addGraph(
        Graph(window, title="Unaltered data", rawXData=xVals, rawYData=yVals,
              yLabel="Amplitude (px)", xLabel="Time (s)"), plot=False)
    fit = window.addGraph(unaltered.getCurveFit(window.quadratic), parent=unaltered, plot=False)
    driftRm = window.addGraph(unaltered - fit, plot=False)
    driftRm.setTitle("Drift Removed")
    unitConverted = window.addGraph(driftRm.convertUnits(yMultiplier=1.0 / 142857.0, yLabel="Position (rad)"),
                                    plot=False)
    window.addGraph(unitConverted.slice(0, 1000))


if __name__ == "__main__":
    main = MainWindow()
    #generateEQPGraphs(main)
    generateTestGraphs(main)
    main.mainloop()
