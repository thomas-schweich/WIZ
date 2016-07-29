from functools import partial
import Tkinter as Tk


class GraphSelector(Tk.Frame):

    def __init__(self, rootWindow, graphsInAxis):
        """Creates a window prompting the user to select a graph from the axis if len(graphsInAxis) > 1

        otherwise opens the graph's GraphWindow"""
        Tk.Frame.__init__(self, rootWindow)
        self.rootWindow = rootWindow
        self.graphsInAxis = [gr for gr in graphsInAxis if gr.isShown()]
        print "Graphs in axis: %s" % str(graphsInAxis)
        self.radioVar = Tk.IntVar(self.rootWindow)
        self.window = None

    def populate(self):
        if len(self.graphsInAxis) > 1:
            self.window = Tk.Toplevel(self)
            #self.window = Tk.Toplevel(self.rootWindow)
            Tk.Label(self.window, text="Available Graphs on this axis:").pack()
            for i, graph in enumerate(self.graphsInAxis):
                frame = Tk.Frame(self.window)
                frame.pack(fill=Tk.X)
                Tk.Button(frame, text=str(graph.title),
                          command=partial(self.openGrWinFromDialogue, graph, self.window)).pack(fill=Tk.X, expand=True,
                                                                                                side=Tk.LEFT)
                radiobutton = Tk.Radiobutton(frame, variable=self.radioVar, value=i,
                                             command=partial(self.setMaster, graph, self.graphsInAxis))
                if graph.master:
                    radiobutton.select()
                else:
                    radiobutton.deselect()
                radiobutton.pack(side=Tk.RIGHT)
        else:
            self.graphsInAxis[0].openWindow()

    def setMaster(self, graph, graphsInAxis):
        print "setMaster() called"
        for g in graphsInAxis:
            if g.master:
                print "%s was master" % str(g)
            g.master = False
        graph.master = True
        self.rootWindow.plotGraphs()

    @staticmethod
    def openGrWinFromDialogue(graph, window):
        """Opens graph's GraphWindow and destroys window"""
        graph.openWindow()
        window.destroy()
