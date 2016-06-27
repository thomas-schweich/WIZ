import matplotlib

matplotlib.use("TkAgg")
import math
import sys

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from Graph import Graph
from functools import partial
from copy import copy

__author__ = "Thomas Schweich"


class MainWindow(Tk.Tk):
    def __init__(self, graphs=None, *args, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        Tk.Tk.__init__(self, *args, **kwargs)
        if not graphs: graphs = []
        plt.style.use("ggplot")
        self.wm_title("Data Manipulation")
        self.defaultWidth, self.defaultHeight = self.winfo_screenwidth(), self.winfo_screenheight() * .9
        self.geometry("%dx%d+0+0" % (self.defaultWidth, self.defaultHeight))
        self.graphs = graphs
        self.buttons = []
        self.topFrame = Tk.Frame(self)
        self.topFrame.pack(side=Tk.TOP, fill=Tk.X)
        self.buttonFrame = Tk.Frame(self.topFrame)
        self.buttonFrame.pack(side=Tk.TOP, expand=1, fill=Tk.X)
        self.f = Figure(figsize=(5, 4), dpi=150)
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", self.onClick)
        self.generateEQPGraphs()

    def _quit(self):
        """Closes the MainWindow"""
        self.root.quit()
        self.root.destroy()

    @staticmethod
    def cleanData(path):
        """Removes NaNs from the data stored at the path and returns the resulting numpy array"""
        xData, yData = np.loadtxt(path, unpack=True, dtype=float)
        xNans = np.isnan(xData)
        yNans = np.isnan(yData)
        nans = np.logical_or(xNans, yNans)
        notNans = np.logical_not(nans)
        xData = xData[notNans]
        yData = yData[notNans]
        return xData, yData

    def addGraph(self, graph, parent=None, plot=True):
        """Adds a graph to this MainWindow's .graphs list, plotting it unless plot is set to false"""
        self.addGraphToAxisList(self.graphs, graph, parent=parent)
        if plot:
            self.plotGraphs()
        return graph

    @staticmethod
    def addGraphToAxisList(axisList, graph, parent=None):
        """Takes a list and adds a graph to it, with location depending on its parent"""
        if not parent:
            axisList.append([graph])
        else:
            for axis in axisList:
                if len(axis) > 0:
                    for g in axis:
                        if g is parent:
                            axis.append(graph)

    def removeGraph(self, graph, plot=True):
        """Removes the graph from the MainWindow's .graphs list, re-plotting unless plot is False"""
        for axis in self.graphs:
            if graph in axis:
                axis.remove(graph)
        if plot: self.plotGraphs()

    def onClick(self, event):
        """If event.dblclick, calls the MainWindow's promptSelect() method with the axis designated by event.inaxis"""
        if event.dblclick:
            for axis in self.graphs:
                if event.inaxes is axis[0].subplot:
                    self.promptSelect(axis)
                    return

    def promptSelect(self, graphsInAxis):
        """Creates a window prompting the user to select a graph from the axis if len(graphsInAxis) > 1

        otherwise opens the graph's GraphWindow"""
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
        """Plots all graphs in the MainWindows .graphs list, creating a button for each"""
        self.f.clear()
        self.clearButtons()
        graphsToShow = copy(self.graphs)
        for axis in graphsToShow:
            for graph in axis:
                self.buttons.append(Tk.Button(self.buttonFrame, text=str(graph.title), command=graph.openWindow))
                if not graph.isShown():
                    axis.remove(graph)
            if len(axis) < 1:
                graphsToShow.remove(axis)
        length = len(graphsToShow)
        rows = math.ceil(length / 2.0)
        subplots = [self.f.add_subplot(rows, 2, index + 1) for index in range(0, length)]
        for idx, axis in enumerate(graphsToShow):
            for g in axis:
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

    def generateEQPGraphs(self):
        """Sample method for generating default graphs in a chain"""
        xVals, yVals = self.cleanData("BigEQPTest.txt")
        unaltered = self.addGraph(
            Graph(self, title="Unaltered data", rawXData=xVals, rawYData=yVals, autoScaleMagnitude=False,
                  yLabel="Amplitude (px)", xLabel="Time (s)", root=self), plot=False)
        fit = self.addGraph(unaltered.getCurveFit(self.quadratic), parent=unaltered, plot=False)
        driftRm = self.addGraph(unaltered - fit, plot=False)
        driftRm.setTitle("Drift Removed")
        unitConverted = self.addGraph(driftRm.convertUnits(yMultiplier=1.0 / 142857.0, yLabel="Position (rad)"),
                                      plot=False)
        self.addGraph(unitConverted.slice(begin=60000, end=100000))


if __name__ == "__main__":
    main = MainWindow()
    main.mainloop()
