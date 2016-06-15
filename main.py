import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from Graph import Graph
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.widgets import Button

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

__author__ = "Thomas Schweich"


def _quit():
    root.quit()
    root.destroy()


def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)


def cleanData(path):
    xData, yData = np.loadtxt(path, unpack=True, dtype=float)
    xNans = np.isnan(xData)
    yNans = np.isnan(yData)
    nans = np.logical_or(xNans, yNans)
    notNans = np.logical_not(nans)
    xData = xData[notNans]
    yData = yData[notNans]
    return xData, yData


def quadratic(x, a, b, c):
    return a*x**2 + b*x + c


def sinusoid(x, a, b, c, d):
    return a*(np.sin(b*x + c)) + d


plt.style.use("ggplot")

root = Tk.Tk()
root.wm_title("Data Manipulation")

f = Figure(figsize=(5, 4), dpi=150)

xVals, yVals = cleanData("BigEQPTest.txt")
unaltered = Graph(title="Unaltered data", rawXData=xVals, rawYData=yVals, autoScaleMagnitude=False,
                  yLabel="Amplitude (px)", xLabel="Time (s)")
unalteredSub = f.add_subplot(221)
unaltered.setSubplot(unalteredSub)
unaltered.plot()
fit = unaltered.getCurveFit(quadratic)
fit.setTitle("Fit")
fit.setSubplot(unalteredSub)
fit.plot()
driftRm = unaltered-fit
driftRm.setTitle("Drift Removed")
driftRmSub = f.add_subplot(222)
driftRm.plot(driftRmSub)
unitConverted = driftRm.convertUnits(yMultiplier=1.0/142857.0, yLabel="Position (rad)")
unitConvertedSub = f.add_subplot(223)
unitConverted.plot(unitConvertedSub)
subsection = unitConverted.slice(begin=60000, end=100000)
subSectionSub = f.add_subplot(224)
subsection.plot(subSectionSub)

canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

canvas.mpl_connect('key_press_event', on_key_event)

#button = Tk.Button(master=root, text='Quit', command=_quit)
#button.pack(side=Tk.BOTTOM)

cid = f.canvas.mpl_connect('button_press_event', unaltered.onClick)
'''
axes = unalteredSub.get_axes()
axesButton = Button(axes, "")
axesButton.on_clicked(unaltered.onClick)
'''
Tk.mainloop()
