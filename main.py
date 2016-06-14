import matplotlib.pyplot as plt
import numpy as np
from Graph import Graph


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

xVals, yVals = cleanData("BigEQPTest.txt")
unaltered = Graph(title="Unaltered data", rawXData=xVals, rawYData=yVals, autoScaleMagnitude=False,
                  xLabel="Amplitude (px)", yLabel="Time (s)")
fit = unaltered.getCurveFit(quadratic)
fit.setTitle("Fit")
plt.subplot(221)
fit.plot()
unaltered.plot()
driftRm = unaltered-fit
driftRm.setTitle("Drift Removed")
plt.subplot(222)
driftRm.plot()
unitConverted = driftRm.convertUnits(yMultiplier=1.0/142857.0, yLabel="Position (rad)")
plt.subplot(223)
unitConverted.plot()
subsection = unitConverted.slice(begin=60000, end=100000)
plt.subplot(224)
subsection.plot()
plt.show()
