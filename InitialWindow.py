import Tkinter as Tk
import ttk
from Graph import Graph
from MainWindow import MainWindow
import tkFileDialog
import numpy as np


class InitialWindow(Tk.Tk):
    """Tk object which creates an initial window for loading projects and files, ultimately creating a MainWindow"""

    def __init__(self, *args, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        Tk.Tk.__init__(self, *args, **kwargs)
        self.wm_title("GEE Data Manipulator")
        self.defaultWidth, self.defaultHeight = self.winfo_screenwidth() * .25, self.winfo_screenheight() * .25
        self.geometry("%dx%d+0+0" % (self.defaultWidth, self.defaultHeight))
        self.baseFrame = Tk.Frame(master=self)
        self.baseFrame.pack()
        text = Tk.Label(self.baseFrame, text="Welcome to the GEE Data Manipulator")
        text.pack()
        loadButton = Tk.Button(self.baseFrame, text="Load Project", command=self.loadProject)
        loadButton.pack(fill=Tk.X)
        rawButton = Tk.Button(self.baseFrame, text="Load Raw Data", command=self.loadRawData)
        rawButton.pack(fill=Tk.X)
        blankButton = Tk.Button(self.baseFrame, text="New Blank Project", command=self.createBlankProject)
        blankButton.pack(fill=Tk.X)

    def loadProject(self):
        """Loads an .npz file using MainWindow.loadProject"""
        path = tkFileDialog.askopenfilename()
        self.quit()
        self.destroy()
        MainWindow.loadProject(path)

    def loadRawData(self):
        """Loads data from a text file using MainWindow.loadData and prompts the user for a slice"""
        newFrame = Tk.Frame(self.baseFrame)
        newFrame.pack(side=Tk.BOTTOM)
        path = tkFileDialog.askopenfilename()
        # loading = ttk.Progressbar(newFrame)
        # loading.pack()
        # loading.start(interval=10)
        data = MainWindow.loadData(path)
        # loading.stop()
        # loading.destroy()
        Tk.Label(newFrame, text="How much data would you like to use?").pack()
        tkVar = Tk.IntVar()
        start = Tk.Entry(newFrame)
        start.insert(0, "0")
        end = Tk.Entry(newFrame)
        end.insert(0, str(len(data[0])))
        start.pack()
        end.pack()
        Tk.Radiobutton(newFrame, text="By Index (from 0 to %d)" % len(data[0]), variable=tkVar, value=0).pack()
        Tk.Radiobutton(newFrame, text="By x-value (from %d to %d)" % (data[0][0], data[0][-1]), variable=tkVar,
                       value=1).pack()
        Tk.Button(newFrame, text="Create Project",
                  command=lambda: self.sliceData(data, tkVar, start.get(), end.get())).pack()

    def createBlankProject(self):
        self.quit()
        self.destroy()
        win = MainWindow()
        win.setGraphs([[Graph(window=win, title="New Project")]])
        win.plotGraphs()
        win.mainloop()

    def sliceData(self, data, tkVar, begin, end):
        """Creates a graph of the slice of data and creates a new MainWindow with .graphs assigned to the new graph"""
        begin = float(begin)
        end = float(end)
        # By index
        if tkVar.get() == 0:
            newDat = data[0][begin:end], data[1][begin:end]
        # By x value
        else:  # elif tkVar.get() == 1:
            newBegin, newEnd = \
                np.searchsorted(data[0], np.array([np.float64(begin), np.float64(end)]))
            newDat = data[0][newBegin:newEnd], data[1][newBegin, newEnd]
        self.quit()
        self.destroy()
        win = MainWindow()
        gr = Graph(window=win, title="Raw Data")
        gr.setRawData(newDat)
        win.setGraphs([[gr]])
        win.plotGraphs()
        win.mainloop()

if __name__ == "__main__":
    initial = InitialWindow()
    initial.mainloop()
