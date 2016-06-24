import matplotlib

matplotlib.use('TkAgg')
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
    def __init__(self, *args, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        Tk.Tk.__init__(self, *args, **kwargs)
        plt.style.use("ggplot")
        self.wm_title("Data Manipulation")
        self.graphs = []
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
        self.canvas.mpl_connect('button_press_event', self.onClick)
        self.generateEQPGraphs()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    @staticmethod
    def cleanData(path):
        xData, yData = np.loadtxt(path, unpack=True, dtype=float)
        xNans = np.isnan(xData)
        yNans = np.isnan(yData)
        nans = np.logical_or(xNans, yNans)
        notNans = np.logical_not(nans)
        xData = xData[notNans]
        yData = yData[notNans]
        return xData, yData

    def addGraph(self, graph, parent=None):
        if not parent:
            self.graphs.append([graph])
            # self.f.canvas.mpl_connect('button_press_event', graph.onClick)
        else:
            for axis in self.graphs:
                print axis
                if len(axis) > 0:
                    for g in axis:
                        if g is parent:
                            axis.append(graph)
        # self.plotGraphs()
        # self.f.canvas.mpl_connect('button_press_event', graph.onClick)  # Moving to a single connect which calls all
        #  graphs in graph list
        self.plotGraphs()
        return graph

    def removeGraph(self, graph):
        self.graphs.remove(graph)
        # self.plotGraphs()

    def onClick(self, event):
        if event.dblclick:
            for axis in self.graphs:
                if event.inaxes is axis[0].subplot:
                    self.promptSelect(axis)
                    return

    def promptSelect(self, graphsInAxis):
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
        graph.openWindow()
        window.destroy()

    def plotGraphs(self):
        '''
        existingSubGraphs = [graph for graph in self.graphs if graph.show and graph.subplot]
        graphsToSubplot = [graph for graph in self.graphs if graph.show]
        self.f.clear()
        orderedGraphs = []
        i = 0
        for gr in graphsToSubplot:
            preexistingPlot = False
            for sub in existingSubGraphs:
                if gr.subplot is sub.subplot and gr is not sub:
                    childList = []
                    for x in orderedGraphs:
                        if sub in x:
                            childList = x
                    if len(childList) > 0:
                        orderedGraphs[orderedGraphs.index(childList)].append(gr)
                        preexistingPlot = True
                        i -= 1
            if not preexistingPlot:
                orderedGraphs.append([gr])
            i += 1
        '''
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
        subplots = []
        for index in range(0, length):
            subplots.append(self.f.add_subplot(rows, 2, index + 1))

        for idx, ordered in enumerate(graphsToShow):
            for g in ordered:
                g.setSubplot(subplots[idx])
                g.plot()

        for button in self.buttons:
            button.pack(side=Tk.LEFT, fill=Tk.X, expand=1)

    def clearButtons(self):
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
        xVals, yVals = self.cleanData("BigEQPTest.txt")
        unaltered = self.addGraph(
            Graph(self, title="Unaltered data", rawXData=xVals, rawYData=yVals, autoScaleMagnitude=False,
                  yLabel="Amplitude (px)", xLabel="Time (s)", root=self))
        unaltered.setSubplot(1)
        fit = self.addGraph(unaltered.getCurveFit(self.quadratic), parent=unaltered)
        fit.setSubplot(1)
        driftRm = self.addGraph(unaltered - fit)
        driftRm.setTitle("Drift Removed")
        unitConverted = self.addGraph(driftRm.convertUnits(yMultiplier=1.0 / 142857.0, yLabel="Position (rad)"))
        subsection = self.addGraph(unitConverted.slice(begin=60000, end=100000))
        #self.plotGraphs()



if __name__ == "__main__":
    main = MainWindow()
    main.mainloop()
