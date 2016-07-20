import matplotlib

matplotlib.use('TkAgg')
import sys

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import tkFileDialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from copy import copy, deepcopy
from MathExpression import MathExpression
import os
import math
import Graph


class GraphWindow(Tk.Frame):
    """Tk.Frame child which hooks into a Graph to provide it modification options

    A Tk.Toplevel instance is created when open() is called. Only one per GraphWindow instance is allowed at a time.
    """

    __author__ = "Thomas Schweich"

    def __init__(self, graph, *args, **kwargs):
        """A frame object who's open() method creates a Tk.Toplevel (new window) with its contents"""
        Tk.Frame.__init__(self, *args, **kwargs)
        self.widgets = {}
        self.graph = graph
        print "Opening window for %s" % graph.getTitle()
        self.newGraph = None
        self.graphSubPlot = None
        self.newSubPlot = None
        self.window = None
        self.radioVar = Tk.IntVar()
        self.fitBox = None
        self.sliceBox = None
        self.addBox = None
        self.multBox = None
        self.customBox = None
        self.canvas = None
        self.f = None
        self.baseGroup = None
        self.leftGroup = None
        self.dynamicOptionGroup = None
        self.rbFrame = None
        self.optionsFrame = None
        self.graphOptionsFrame = None
        self.TransformationOptionsFrame = None
        self.pack()
        self.isOpen = False

    def open(self):
        """Opens a graph window only if there isn't already one open for this GraphWindow

        Thus only one window per Graph can be open using this method (assuming Graphs only have one GraphWindow)"""
        if not self.isOpen:
            self.isOpen = True
            self.window = Tk.Toplevel(self)
            self.window.wm_title(str(self.graph.getTitle()))
            self.window.geometry("%dx%d+0+0" % (self.graph.window.winfo_width(), self.graph.window.winfo_height()))
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            self.f = Figure(figsize=(2, 1), dpi=150)
            self.graphSubPlot = self.f.add_subplot(121)
            self.graph.plot(subplot=self.graphSubPlot)
            self.newSubPlot = self.f.add_subplot(122)
            self.newGraph = copy(self.graph)
            self.newGraph.setTitle("Transformation of " + str(self.graph.getTitle()))
            self.newGraph.plot(subplot=self.newSubPlot)
            self.canvas = FigureCanvasTkAgg(self.f, self.window)
            self.canvas.draw()
            self.canvas.show()
            self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
            self.baseGroup = Tk.Frame(self.window)
            self.baseGroup.pack(side=Tk.BOTTOM, fill=Tk.BOTH)
            self.leftGroup = Tk.Frame(self.baseGroup)
            self.leftGroup.pack(side=Tk.LEFT)
            self.TransformationOptionsFrame = Tk.Frame(self.leftGroup)
            self.TransformationOptionsFrame.pack(side=Tk.TOP, fill=Tk.BOTH)
            self.graphOptionsFrame = Tk.Frame(self.leftGroup)
            self.graphOptionsFrame.pack(side=Tk.TOP, fill=Tk.BOTH)
            self.dynamicOptionGroup = Tk.Frame(self.baseGroup)
            self.dynamicOptionGroup.pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=1)
            self.optionsFrame = Tk.Frame(self.dynamicOptionGroup)
            self.optionsFrame.pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)
            self.rbFrame = Tk.Frame(self.dynamicOptionGroup)
            self.rbFrame.pack(side=Tk.TOP, fill=Tk.BOTH)
            self.populate()
            self.refreshOptions()
            self.canvas.mpl_connect("button_press_event", lambda event: self.onClick(event))
            # self.f.draw()

    def close(self):
        """Destroys the window, sets the GraphWindows's Toplevel instance to None"""
        del self.widgets
        if self.f:
            self.f.clf()
            plt.close(self.f)
            del self.f
        self.widgets = {}
        self.window.destroy()
        self.isOpen = False

    def populate(self):
        """Adds all widgets to the window in their proper frames, with proper cascading"""
        # BASE OPTIONS
        Tk.Label(self.TransformationOptionsFrame, text="Transformation Options").pack(fill=Tk.X)
        Tk.Button(self.TransformationOptionsFrame, text="Plot on This Axis", command=self.plotOnThisAxis).pack(
            fill=Tk.X)
        Tk.Button(self.TransformationOptionsFrame, text="Plot on New Axis", command=self.plotOnNewAxis).pack(fill=Tk.X)
        Tk.Button(self.TransformationOptionsFrame, text="Replace Graph", command=self.replaceGraph).pack(fill=Tk.X)
        Tk.Button(self.TransformationOptionsFrame, text="Cancel", command=self.close).pack(fill=Tk.X)
        Tk.Label(self.graphOptionsFrame, text="Graph Options").pack(fill=Tk.X)
        Tk.Button(self.graphOptionsFrame, text="Save Graph", command=self.saveGraph).pack(fill=Tk.X)
        Tk.Button(self.graphOptionsFrame, text="Save Data", command=self.saveData).pack(fill=Tk.X)
        Tk.Button(self.graphOptionsFrame, text="Delete Graph", command=self.removeGraph).pack(fill=Tk.X)
        showVal = Tk.IntVar()
        showVal.set(self.graph.isShown())
        Tk.Checkbutton(self.graphOptionsFrame, text="Show", variable=showVal, onvalue=1, offvalue=0,
                       command=lambda: self.showHide(showVal)).pack(fill=Tk.X)

        # FIT
        self.fitBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions, text="Fit Options",
                                     variable=self.radioVar, value=0)
        self.fitBox.val = 0
        self.addWidget(Tk.Button, parent=self.fitBox, command=self.quarticFit, text="Quartic Fit")
        self.addWidget(Tk.Button, parent=self.fitBox, command=self.cubicFit, text="Cubic Fit")
        self.addWidget(Tk.Button, parent=self.fitBox, command=self.quadraticFit, text="Quadratic Fit")
        self.addWidget(Tk.Button, parent=self.fitBox, command=self.linearFit, text="Linear Fit")

        # SLICE
        self.sliceBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions, text="Slice Options",
                                       variable=self.radioVar, value=1)
        self.sliceBox.val = 1
        sliceVar = Tk.IntVar()
        self.addWidget(Tk.Radiobutton, parent=self.sliceBox,
                       text="By index (from 0 to " + str(len(self.graph.getRawData()[0])) +
                            ")", variable=sliceVar, value=0)
        self.addWidget(Tk.Radiobutton, parent=self.sliceBox,
                       text="By nearest x value (from " + str(self.graph.getRawData()[0][0]) + " to " + str(
                           self.graph.getRawData()[0][len(self.graph.getRawData()[0]) - 1]) + ")",
                       variable=sliceVar, value=1)

        start = self.addWidget(Tk.Entry, parent=self.sliceBox)
        start.insert(0, "Start")
        end = self.addWidget(Tk.Entry, parent=self.sliceBox)
        end.insert(0, "End")
        self.addWidget(Tk.Button, parent=self.sliceBox, command=lambda: self.addSlice(sliceVar, start.get(), end.get()),
                       text="Preview")

        # ADD
        self.addBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions,
                                     text="Add/Subtract Graphs", variable=self.radioVar, value=2)
        self.addBox.val = 2
        self.addWidget(Tk.Label, parent=self.addBox, text="By constant:")
        addConstant = self.addWidget(Tk.Entry, parent=self.addBox)
        addConstant.insert(0, "Constant")
        self.addWidget(Tk.Button, parent=self.addBox, text="Add with constant",
                       command=lambda: self.addAddition(float(addConstant.get())))
        graphs = [g for a in self.graph.window.graphs for g in a]
        sameXGraphs = [g for g in graphs if g.isSameX(self.graph)]
        graphTitles = tuple([g.getTitle() for g in sameXGraphs])
        addDropVar = Tk.StringVar()
        addDropVar.set(graphTitles[0] if len(graphTitles) > 0 else "")
        self.addWidget(Tk.Label, parent=self.addBox, text="By Graph:")
        addDropdown = Tk.OptionMenu(self.optionsFrame, addDropVar, *graphTitles)
        self.widgets[self.addBox].append(addDropdown)  # Manual addition
        self.addWidget(Tk.Button, parent=self.addBox, text="Add Graphs",
                       command=lambda: self.addAddition(sameXGraphs[graphTitles.index(addDropVar.get())]))
        self.addWidget(Tk.Button, parent=self.addBox, text="Subtract Graphs",
                       command=lambda: self.addSubtraction(sameXGraphs[graphTitles.index(addDropVar.get())]))

        # MULTIPLY
        self.multBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions,
                                      text="Multiply/Divide", variable=self.radioVar, value=3)
        self.multBox.val = 3
        self.addWidget(Tk.Label, parent=self.multBox, text="By constant:")
        multConstant = self.addWidget(Tk.Entry, parent=self.multBox)
        multConstant.insert(0, "Constant")
        self.addWidget(Tk.Button, parent=self.multBox, text="Multiply with constant",
                       command=lambda: self.addMultiplication(float(multConstant.get())))
        multDropVar = Tk.StringVar()
        multDropVar.set(graphTitles[0] if len(graphTitles) > 0 else "")
        self.addWidget(Tk.Label, parent=self.multBox, text="By graph:")
        multDropdown = Tk.OptionMenu(self.optionsFrame, multDropVar, *graphTitles)
        self.widgets[self.multBox].append(multDropdown)  # Manual addition
        self.addWidget(Tk.Button, parent=self.multBox, text="Multiply Graphs",
                       command=lambda: self.addMultiplication(sameXGraphs[graphTitles.index(multDropVar.get())]))
        self.addWidget(Tk.Button, parent=self.multBox, text="Divide Graphs",
                       command=lambda: self.addDivision(sameXGraphs[graphTitles.index(addDropVar.get())]))

        # CUSTOM EXPRESSION
        self.customBox = self.addWidget(Tk.Radiobutton, command=self.refreshOptions, text="Custom Expression",
                                        variable=self.radioVar, value=4)
        self.customBox.val = 4

        customGraphTitles = tuple([g.getTitle() for g in graphs])
        textBox = self.addWidget(Tk.Text, parent=self.customBox, font=("Mono", 15))
        textBox.insert(Tk.END, "<%s>" % str(self.graph.getTitle()))
        customDropVar = Tk.StringVar()
        customDropVar.set(customGraphTitles[0] if len(customGraphTitles) > 0 else "")
        self.addWidget(Tk.Label, parent=self.customBox, text="Reference y-data of graph:")
        customDropdown = Tk.OptionMenu(self.optionsFrame, customDropVar, *customGraphTitles, command=lambda name:
                       textBox.insert(Tk.INSERT, "<%s>" % str(customGraphTitles[customGraphTitles.index(name)])))
        self.widgets[self.customBox].append(customDropdown)  # Manual addition
        self.addWidget(Tk.Button, parent=self.customBox, text="Parse",
                       command=lambda: self.parseExpression(textBox.get(1.0, Tk.END)))

        # TODO Cases with multiple graphs of the same title
        # TODO Cases without matching graphs

    def refreshOptions(self):
        """Refreshes the displayed options based on the currently selected Radiobutton"""
        for k, v in self.widgets.iteritems():
            for widget in v:
                if widget.winfo_exists():
                    widget.pack_forget()
            # Main Radiobuttons
            try:
                if k.val == self.radioVar.get():
                    for widget in v:
                        widget.pack(side=Tk.TOP, expand=0)
            except AttributeError:
                for widget in v:
                    widget.pack(side=Tk.BOTTOM, expand=1)
        # TODO Make more dynamic

    def onClick(self, event):
        if event.dblclick:
            if event.inaxes is self.newSubPlot:
                self.openOptions()

    def openOptions(self):
        window = Tk.Toplevel(self)
        frame = Tk.Frame(window)
        frame.pack()
        Tk.Label(frame, text="Options").pack()
        Tk.Label(frame, text="Title:").pack()
        titleEntry = Tk.Entry(frame)
        titleEntry.insert(0, self.graph.getTitle())
        titleEntry.pack()
        Tk.Label(frame, text="X-Label:").pack()
        xLabelEntry = Tk.Entry(frame)
        xLabelEntry.insert(0, self.graph.xLabel)
        xLabelEntry.pack()
        Tk.Label(frame, text="Y-Label:").pack()
        yLabelEntry = Tk.Entry(frame)
        yLabelEntry.insert(0, self.graph.yLabel)
        yLabelEntry.pack()
        applyButton = Tk.Button(frame, text="Apply", command=lambda: self.setLabels(
            window, titleEntry.get(), xLabelEntry.get(), yLabelEntry.get()))
        applyButton.pack()

    def setLabels(self, window, title, xLabel, yLabel):
        self.newGraph.setTitle(title)
        self.newGraph.setXLabel(xLabel)
        self.newGraph.setYLabel(yLabel)
        self.f.delaxes(self.newSubPlot)
        self.newSubPlot = self.f.add_subplot(122)
        self.newGraph.plot(subplot=self.newSubPlot)
        self.canvas.show()
        window.quit()
        window.destroy()

    def addWidget(self, widgetType, parent=None, *args, **kwargs):
        """Adds a widget to the window.

        If a parent is specified, it will be placed in the widgets widgets dict under its parent.
        If no parent is specified, it will get its own entry in widgets, and immediately be packed.
        If a parent is specified which doesn't already have its own entry, one will be created for it. Thus, cascading
        trees of widgets can be created. A widget may be both a parent and a child.
        All values are added in lists so that a parent widget may have more than one child.
        """
        if not parent:
            wid = widgetType(self.rbFrame, *args, **kwargs)
            self.widgets[wid] = []
            wid.pack(expand=True, side=Tk.LEFT)
        else:
            wid = widgetType(self.optionsFrame, *args, **kwargs)
            if self.widgets[parent] in self.widgets.values():
                self.widgets[parent].append(wid)
            else:
                self.widgets[parent] = [wid]
        return wid

    def plotWithReference(self, graph):
        """Plots the graph in the window while maintaining a copy of the original graph on the same axes"""
        self.f.delaxes(self.newSubPlot)
        self.newSubPlot = self.f.add_subplot(122)
        referenceGraph = copy(self.graph)
        self.newGraph = graph
        referenceGraph.plot(subplot=self.newSubPlot)
        self.newGraph.plot(subplot=self.newSubPlot)
        self.canvas.show()

    def plotAlone(self, graph):
        """Replaces the plot of the original graph in the new graph window with the graph specified"""
        self.f.delaxes(self.newSubPlot)
        self.newSubPlot = self.f.add_subplot(122)
        self.newGraph = graph
        self.newGraph.plot(subplot=self.newSubPlot)
        self.canvas.show()

    def plotOnThisAxis(self):
        """Adds the transformation Graph to the axis of this GraphWindow's Graph"""
        self.graph.window.addGraph(self.newGraph, parent=self.graph)
        self.close()

    def plotOnNewAxis(self):
        """Adds the transformation Graph to a new axis in the Graph's MainWindow"""
        self.graph.window.addGraph(self.newGraph)
        self.close()

    def replaceGraph(self):
        """Places the new graph on the axis of .graph and hides .graph"""
        '''
        from Graph import Graph
        newGraph = Graph(self.graph.window)
        setattr(newGraph, 'rawXData', self.newGraph.rawXData)
        setattr(newGraph, 'rawYData', self.newGraph.rawYData)
        for att in self.newGraph.getMetaData():
            setattr(newGraph, att, self.newGraph.getMetaData()[att])
        '''
        self.graph.window.addGraph(self.newGraph, parent=self.graph, plot=False)
        self.graph.window.removeGraph(self.graph)
        self.close()

    def quarticFit(self):
        """Plots a quartic fit of the Graph's data with reference"""
        self.plotWithReference(self.graph.getCurveFit(
            fitFunction=lambda x, a, b, c, d, e: a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e))

    def cubicFit(self):
        """Plots a cubic fit of the Graph's data with reference"""
        self.plotWithReference(
            self.graph.getCurveFit(fitFunction=lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d))

    def quadraticFit(self):
        """Plots a quadratic fit of the Graph's data with reference"""
        self.plotWithReference(self.graph.getCurveFit(fitFunction=lambda x, a, b, c: a * x ** 2 + b * x + c))

    def linearFit(self):
        """Plots a linear fit of the .graph data with reference"""
        self.plotWithReference(self.graph.getCurveFit(fitFunction=lambda x, a, b: a * x + b))

    def addSlice(self, tkVar, begin, end):
        """Plots a slice of .graph alone"""
        # By index
        if tkVar.get() == 0:
            self.plotAlone(self.graph.slice(begin=float(begin), end=float(end)))
        # By x value
        elif tkVar.get() == 1:
            results = np.searchsorted(self.graph.getRawData()[0], np.array([np.float64(begin), np.float64(end)]))
            self.plotAlone(self.graph.slice(begin=results[0], end=results[1]))

    def addAddition(self, val):
        """Plots a Graph of .graph + val alone"""
        self.plotAlone(self.graph + val)

    def addSubtraction(self, val):
        """Plots a Graph of .graph - val alone"""
        self.plotAlone(self.graph - val)

    def addMultiplication(self, val):
        """Plots a Graph of .graph * val alone"""
        self.plotAlone(self.graph * val)

    def addDivision(self, val):
        """Plots a Graph of .graph / val alone"""
        self.plotAlone(self.graph / val)

    def removeGraph(self):
        """Removes .graph from its window"""
        self.graph.window.removeGraph(self.graph)
        self.close()

    def saveGraph(self):
        """Saves a pdf, png, or svg of .graph to a user specified directory"""
        fig, ax = plt.subplots()
        self.graph.plot(subplot=ax)
        path = tkFileDialog.asksaveasfilename(defaultextension=".pdf",
                                              filetypes=[("Portable Document Format", ".pdf"),
                                                         ("Portable Network Graphics", ".png"),
                                                         ("Scalable Vector Graphics", ".svg")])
        fig.savefig(path)

    def saveData(self):
        """Saves a csv or npy of .graph's data to a user specified directory

        If any file extension other than .npy is specified, the data will actually be saved in csv format"""
        path = tkFileDialog.asksaveasfilename(defaultextension=".csv",
                                              filetypes=[("Comma Separated Values", ".csv"),
                                                         ("NumPy Array", ".npy")])
        ftype = path[path.rfind("."):]
        if ftype == ".npy":
            np.save(path, self.graph.getRawData())
        else:
            np.savetxt(path, np.dstack(self.graph.getRawData())[0], delimiter=",")

    def showHide(self, checkVal):
        print checkVal.get()
        if checkVal.get() == 0:
            self.graph.show = False
        if checkVal.get() == 1:
            self.graph.show = True
        self.graph.window.plotGraphs()

    def parseExpression(self, expression):
        graphVars = {}
        for axis in self.graph.window.graphs:
            for graph in axis:
                graphVars[graph.getTitle()] = copy(graph)
        print graphVars
        exp = MathExpression(str(expression), modules=(Graph, np, math), variables=graphVars, fallbackFunc=self.graph.useYForCall)
        graph = exp.evaluate()
        self.plotAlone(graph)

    @staticmethod
    def avoidDuplicates(path, getExtension=False):
        i = 1
        ptIndex = path.rfind(".")
        while os.path.isfile(path):
            startindex = ptIndex - (2 + math.ceil(math.log10(i)))
            end = path[startindex:ptIndex]
            if end == "(" + str(i - 1) + ")":
                path = path[:startindex] + path[ptIndex:]
            path = path[:ptIndex] + "(" + str(i) + ")" + path[ptIndex:]
            i += 1
        if getExtension:
            ftype = path[ptIndex:]
            return path, ftype
        else:
            return path
