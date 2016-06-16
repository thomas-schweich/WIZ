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


class MainWindow(Tk.Frame):
    def __init__(self, *args, **kwargs):
        Tk.Frame.__init__(self, *args, **kwargs)
        plt.style.use("ggplot")
        self.root = args[0]
        self.root.wm_title("Data Manipulation")

        f = Figure(figsize=(5, 4), dpi=150)

        xVals, yVals = self.cleanData("BigEQPTest.txt")
        unaltered = Graph(title="Unaltered data", rawXData=xVals, rawYData=yVals, autoScaleMagnitude=False,
                          yLabel="Amplitude (px)", xLabel="Time (s)")
        unalteredSub = f.add_subplot(221)
        unaltered.setSubplot(unalteredSub)
        unaltered.plot()
        fit = unaltered.getCurveFit(self.quadratic)
        fit.setTitle("Fit")
        fit.setSubplot(unalteredSub)
        fit.plot()
        driftRm = unaltered - fit
        driftRm.setTitle("Drift Removed")
        driftRmSub = f.add_subplot(222)
        driftRm.plot(driftRmSub)
        unitConverted = driftRm.convertUnits(yMultiplier=1.0 / 142857.0, yLabel="Position (rad)")
        unitConvertedSub = f.add_subplot(223)
        unitConverted.plot(unitConvertedSub)
        subsection = unitConverted.slice(begin=60000, end=100000)
        subSectionSub = f.add_subplot(224)
        subsection.plot(subSectionSub)

        self.canvas = FigureCanvasTkAgg(f, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.canvas.mpl_connect('key_press_event', self.on_key_event)

        # button = Tk.Button(master=root, text='Quit', command=_quit)
        # button.pack(side=Tk.BOTTOM)

        self.UnalteredOnClickCid = f.canvas.mpl_connect('button_press_event', unaltered.onClick)
        Tk.mainloop()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    def on_key_event(self, event):
        print('you pressed %s' % event.key)
        key_press_handler(event, self.canvas, self.toolbar)

    def cleanData(self, path):
        xData, yData = np.loadtxt(path, unpack=True, dtype=float)
        xNans = np.isnan(xData)
        yNans = np.isnan(yData)
        nans = np.logical_or(xNans, yNans)
        notNans = np.logical_not(nans)
        xData = xData[notNans]
        yData = yData[notNans]
        return xData, yData

    def quadratic(self, x, a, b, c):
        return a * x ** 2 + b * x + c

    def sinusoid(self, x, a, b, c, d):
        return a * (np.sin(b * x + c)) + d


if __name__ == "__main__":
    root = Tk.Tk()
    main = MainWindow(root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()

