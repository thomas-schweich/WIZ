'''
xMagnitude = int(math.floor(math.log10(xVals[0])))
yMagnitude = int(math.floor(math.log10(yVals[0])))

xScale = 10**xMagnitude
yScale = 10**yMagnitude

scaledXVals = xVals/xScale
scaledYVals = yVals/yScale

abbrX = scaledXVals[::5]
abbrY = scaledYVals[::5]

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

fitParams, fitCoVariances = curve_fit(quadratic, abbrX, abbrY)
fit = quadratic(scaledXVals, fitParams[0], fitParams[1], fitParams[2])
print "1 standard deviation errors: " + str(np.sqrt(np.diag(fitCoVariances)))


plt.subplot(2, 1, 1)
plt.plot(xVals, yVals)
plt.plot(xVals, fit*yScale)
plt.xlabel("x10^%d" % xMagnitude)
plt.ylabel("x10^%d" % yMagnitude)
#plt.ylim(yVals[0], yVals[len(yVals)-1])
#plt.xlim(xVals[0], xVals[len(xVals)-1])

plt.subplot(2, 1, 2)
driftRm = scaledYVals-fit
plt.plot(scaledXVals, driftRm)
'''