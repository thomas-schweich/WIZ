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
plt.subplot(221)
fit.plot()
unaltered.plot()
driftRmDat = unaltered.getRawData()[1] - fit.getRawData()[1]
driftRm = Graph(title="Drift Removed", rawXData=unaltered.getRawData()[0], rawYData=driftRmDat,
                autoScaleMagnitude=False)
plt.subplot(222)
driftRm.plot()
unitConverted = Graph(title="Unit Converted Data", rawXData=driftRm.getRawData()[0],
                      rawYData=driftRm.getRawData()[1]/14287, yLabel="Position (rad)", xLabel="Time (s)")
plt.subplot(223)
unitConverted.plot()
subsection = Graph(title="Sliced graph", rawXData=unitConverted.getRawData()[0][6000:100000],
                   rawYData=unitConverted.getRawData()[1][6000:100000])
plt.subplot(224)
subsection.plot()
plt.show()
