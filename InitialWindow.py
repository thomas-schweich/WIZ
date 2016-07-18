import Tkinter as Tk
import ttk
from Graph import Graph
from MainWindow import MainWindow
import tkFileDialog
import numpy as np
import json


class InitialWindow(Tk.Tk):
    """Tk object which creates an initial window for loading projects and files, ultimately creating a MainWindow"""

    def __init__(self, *args, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        Tk.Tk.__init__(self, *args, **kwargs)
        with open('programSettings.json', 'r') as settingsFile:
            self.settings = json.load(settingsFile)
        self.wm_title("WIZ")
        self.defaultWidth, self.defaultHeight = self.winfo_screenwidth() * .25, self.winfo_screenheight() * .25
        self.geometry("%dx%d+%d+%d" % (self.defaultWidth, self.defaultHeight, self.defaultWidth * 1.5,
                                       self.defaultHeight))
        self.baseFrame = Tk.Frame(master=self)
        self.baseFrame.pack(fill=Tk.X)
        self.newFrame = Tk.Frame(self.baseFrame)
        self.error = Tk.Label(self.baseFrame, text="Invalid selection", fg="red")
        text = Tk.Label(self.baseFrame, text="Welcome to WIZ")
        text.pack()
        loadButton = ttk.Button(self.baseFrame, text="Load Project", command=self.loadProject)
        loadButton.pack(fill=Tk.X)
        rawButton = ttk.Button(self.baseFrame, text="Load Raw Data", command=self.loadRawData)
        rawButton.pack(fill=Tk.X)
        blankButton = ttk.Button(self.baseFrame, text="New Blank Project", command=self.createBlankProject)
        blankButton.pack(fill=Tk.X)

    def loadProject(self):
        """Loads an .npz file using MainWindow.loadProject"""
        self.error.pack_forget()
        path = tkFileDialog.askopenfilename(filetypes=[("Numpy Zipped", ".npz")])
        try:
            MainWindow.loadProject(path, destroyTk=self)
            #window.lift()
            #window.mainloop()
        except IOError:
            self.error.pack()

    def loadRawData(self):
        """Loads data from a text file using MainWindow.loadData and prompts the user for a slice"""
        self.error.pack_forget()
        self.newFrame.destroy()
        self.newFrame = Tk.Frame(self.baseFrame)
        self.newFrame.pack(side=Tk.BOTTOM)
        path = tkFileDialog.askopenfilename()
        # loading = ttk.Progressbar(newFrame)
        # loading.pack()
        # loading.start(interval=10)
        try:
            data = MainWindow.loadData(path)
        except (ValueError, IOError):
            self.error.pack()
            raise
        # loading.stop()
        # loading.destroy()
        Tk.Label(self.newFrame, text="How much data would you like to use?").pack()
        tkVar = Tk.IntVar()
        start = Tk.Entry(self.newFrame)
        start.insert(0, "0")
        end = Tk.Entry(self.newFrame)
        end.insert(0, str(len(data[0])))
        start.pack()
        end.pack()
        Tk.Radiobutton(self.newFrame, text="By Index (from 0 to %d)" % len(data[0]), variable=tkVar, value=0).pack()
        Tk.Radiobutton(self.newFrame, text="By x-value (from %d to %d)" % (data[0][0], data[0][-1]), variable=tkVar,
                       value=1).pack()
        Tk.Button(self.newFrame, text="Create Project",
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
