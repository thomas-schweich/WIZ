import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from Graph import Graph
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import math
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

__author__ = "Thomas Schweich"


class MainWindow(Tk.Tk):
    def __init__(self, *args, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        Tk.Tk.__init__(self, *args, **kwargs)
        plt.style.use("ggplot")
        self.wm_title("Data Manipulation")

        self.graphs = []

        self.f = Figure(figsize=(5, 4), dpi=150)

        xVals, yVals = self.cleanData("BigEQPTest.txt")
        unaltered = self.addGraph(Graph(title="Unaltered data", rawXData=xVals, rawYData=yVals, autoScaleMagnitude=False,
                          yLabel="Amplitude (px)", xLabel="Time (s)", root=self))
        #unalteredSub = self.f.add_subplot(221)
        unaltered.setSubplot(1)#unalteredSub)
        fit = self.addGraph(unaltered.getCurveFit(self.quadratic))
        fit.setSubplot(1)#unalteredSub)
        driftRm = self.addGraph(unaltered - fit)
        driftRm.setTitle("Drift Removed")
        unitConverted = self.addGraph(driftRm.convertUnits(yMultiplier=1.0 / 142857.0, yLabel="Position (rad)"))
        subsection = self.addGraph(unitConverted.slice(begin=60000, end=100000))
        self.plotGraphs()
        print self.graphs

        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.UnalteredOnClickCid = self.f.canvas.mpl_connect('button_press_event', unaltered.onClick)

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

    def addGraph(self, graph):
        self.graphs.append(graph)
        return graph

    def removeGraph(self, graph):
        self.graphs.remove(graph)

    def plotGraphs(self):
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
        rows = math.ceil(len(orderedGraphs)/2.0)
        subplots = []
        for index in range(0, len(orderedGraphs)):
            subplots.append(self.f.add_subplot(rows, 2, index+1))
        for idx, ordered in enumerate(orderedGraphs):
            for g in ordered:
                g.setSubplot(subplots[idx])
                g.plot()

    @staticmethod
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def sinusoid(x, a, b, c, d):
        return a * (np.sin(b * x + c)) + d


if __name__ == "__main__":
    main = MainWindow()
    main.mainloop()


