import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from Graph import Graph
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

__author__ = "Thomas Schweich"


class MainWindow(Tk.Tk):
    def __init__(self, *args, **kwargs):
        Tk.Tk.__init__(self, *args, **kwargs)
        plt.style.use("ggplot")
        self.wm_title("Data Manipulation")

        self.graphs = []

        f = Figure(figsize=(5, 4), dpi=150)

        xVals, yVals = self.cleanData("BigEQPTest.txt")
        unaltered = self.addGraph(Graph(title="Unaltered data", rawXData=xVals, rawYData=yVals, autoScaleMagnitude=False,
                          yLabel="Amplitude (px)", xLabel="Time (s)", root=self))
        unalteredSub = f.add_subplot(221)
        unaltered.setSubplot(unalteredSub)
        unaltered.plot()
        fit = self.addGraph(unaltered.getCurveFit(self.quadratic))
        #fit.setTitle("Fit")
        fit.setSubplot(unalteredSub)
        fit.plot()
        driftRm = self.addGraph(unaltered - fit)
        driftRm.setTitle("Drift Removed")
        driftRmSub = f.add_subplot(222)
        driftRm.plot(driftRmSub)
        unitConverted = self.addGraph(driftRm.convertUnits(yMultiplier=1.0 / 142857.0, yLabel="Position (rad)"))
        unitConvertedSub = f.add_subplot(223)
        unitConverted.plot(unitConvertedSub)
        subsection = self.addGraph(unitConverted.slice(begin=60000, end=100000))
        subSectionSub = f.add_subplot(224)
        subsection.plot(subSectionSub)

        print self.graphs

        self.canvas = FigureCanvasTkAgg(f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.UnalteredOnClickCid = f.canvas.mpl_connect('button_press_event', unaltered.onClick)
        #Tk.mainloop()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    def cleanData(self, path):
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

    @staticmethod
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def sinusoid(x, a, b, c, d):
        return a * (np.sin(b * x + c)) + d



if __name__ == "__main__":
    #root = Tk.Tk()
    main = MainWindow()
    #main.pack(side="top", fill="both", expand=True)
    main.mainloop()


