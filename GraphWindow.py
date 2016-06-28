import matplotlib

matplotlib.use('TkAgg')
import sys

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from copy import copy

__author__ = "Thomas Schweich"


class GraphWindow(Tk.Frame):
    def __init__(self, graph, *args, **kwargs):
        """A frame object who's open() method creates a Tk.Toplevel (new window) with its contents"""
        Tk.Frame.__init__(self, *args, **kwargs)
        self.widgets = {}
        self.graph = graph
        self.newGraph = None
        self.newSubPlot = None
        self.window = None
        self.radioVar = Tk.IntVar()
        self.fitBox = None
        self.sliceBox = None
        self.addBox = None
        self.multBox = None
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
            self.window.wm_title(str(self.graph.title))
            self.window.geometry("%dx%d+0+0" % (self.graph.window.winfo_width(), self.graph.window.winfo_height()))
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            self.f = Figure(figsize=(2, 1), dpi=150)
            graphSubPlot = self.f.add_subplot(121)
            self.graph.plot(subplot=graphSubPlot)
            self.newSubPlot = self.f.add_subplot(122)
            self.newGraph = copy(self.graph)
            self.newGraph.setTitle("Transformation of " + str(self.graph.title))
            self.newGraph.plot(subplot=self.newSubPlot)
            self.canvas = FigureCanvasTkAgg(self.f, self.window)
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

    def close(self):
        """Destroys the window, sets the GraphWindows's Toplevel instance to None"""
        del self.widgets
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
        Tk.Button(self.TransformationOptionsFrame, text="Cancel", command=self.close).pack(fill=Tk.X)
        Tk.Label(self.graphOptionsFrame, text="Graph Options").pack(fill=Tk.X)
        Tk.Button(self.graphOptionsFrame, text="Save Graph").pack(fill=Tk.X)
        Tk.Button(self.graphOptionsFrame, text="Save Data").pack(fill=Tk.X)
        Tk.Button(self.graphOptionsFrame, text="Delete Graph").pack(fill=Tk.X)
        Tk.Checkbutton(self.graphOptionsFrame, text="Show").pack(fill=Tk.X)

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
        graphs = [g for a in self.graph.window.graphs for g in a if g.isSameX(self.graph)]
        graphTitles = tuple([g.getTitle() for g in graphs])
        print graphTitles
        addDropVar = Tk.StringVar()
        addDropVar.set(graphTitles[0] if len(graphTitles) > 0 else "")
        self.addWidget(Tk.Label, parent=self.addBox, text="By Graph:")
        addDropdown = Tk.OptionMenu(self.optionsFrame, addDropVar, *graphTitles)
        self.widgets[self.addBox].append(addDropdown)  # Manual addition
        self.addWidget(Tk.Button, parent=self.addBox, text="Add Graphs",
                       command=lambda: self.addAddition(graphs[graphTitles.index(addDropVar.get())]))
        self.addWidget(Tk.Button, parent=self.addBox, text="Subtract Graphs",
                       command=lambda: self.addSubtraction(graphs[graphTitles.index(addDropVar.get())]))

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
                       command=lambda: self.addMultiplication(graphs[graphTitles.index(multDropVar.get())]))
        self.addWidget(Tk.Button, parent=self.multBox, text="Divide Graphs",
                       command=lambda: self.addDivision(graphs[graphTitles.index(addDropVar.get())]))

        # TODO Cases without matching graphs?

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
        print self.widgets[wid if not parent else parent]
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

    def plotOnNewAxis(self):
        """Adds the transformation Graph to a new axis in the Graph's MainWindow"""
        self.graph.window.addGraph(self.newGraph)

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
