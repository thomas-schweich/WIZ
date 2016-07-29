import Tkinter as Tk
import ttk
from Graph import Graph
from MainWindow import MainWindow
import tkFileDialog
import numpy as np
import json
import os
import re


class InitialWindow(Tk.Tk):
    """Tk object which creates an initial window for loading projects and files, ultimately creating a MainWindow"""
    __author__ = "Thomas Schweich"

    defaultProgramSettings = {
        "Load Chunk Size": 100000,
        "Plot Chunk Size": 100000,
        "Max Preview Points": 100000,
        "DPI": 150,
        "Style": ["ggplot"],
        "User Font Size": 14,
        "Icon Location": r'res\WIZ.ico'
    }

    def __init__(self, win=None, *args, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        Tk.Tk.__init__(self, *args, **kwargs)
        # Load settings. If they don't exist, create default settings file.
        if os.path.isfile('programSettings.json'):
            with open('programSettings.json', 'r') as settingsFile:
                self.settings = json.load(settingsFile)
        else:
            with open('programSettings.json', 'w+') as settingsFile:
                json.dump(InitialWindow.defaultProgramSettings, settingsFile)
                self.settings = InitialWindow.defaultProgramSettings
        self.iconbitmap(self.settings["Icon Location"])
        self.wm_title("WIZ")
        self.defaultWidth, self.defaultHeight = self.winfo_screenwidth() * .25, self.winfo_screenheight() * .5
        self.geometry("%dx%d+%d+%d" % (self.defaultWidth, self.defaultHeight, self.defaultWidth * 1.5,
                                       0))
        self.win = win
        self.baseFrame = Tk.Frame(master=self)
        self.baseFrame.pack(fill=Tk.X, side=Tk.TOP)
        self.newFrame = Tk.Frame(self.baseFrame)
        self.error = Tk.Label(self.baseFrame, text="Invalid selection", fg="red")
        text = Tk.Label(self.baseFrame, text="Welcome to WIZ")
        text.pack()
        if not self.win:
            loadButton = ttk.Button(self.baseFrame, text="Load Project", command=self.loadProject)
            loadButton.pack(fill=Tk.X, side=Tk.TOP)
            blankButton = ttk.Button(self.baseFrame, text="New Blank Project", command=self.createBlankProject)
            blankButton.pack(fill=Tk.X, side=Tk.TOP)
        rawButton = ttk.Button(self.baseFrame, text="Load Raw Data", command=self.loadRawData)
        rawButton.pack(fill=Tk.X, side=Tk.TOP)

    def loadProject(self):
        """Loads an .npz file using MainWindow.loadProject"""
        self.error.pack_forget()
        path = tkFileDialog.askopenfilename(filetypes=[("WIZ Project", ".gee.npy")])
        loading = Tk.Label(self.baseFrame, text="Loading...")
        loading.pack()
        self.update()
        try:
            MainWindow.loadProject(path, destroyTk=self)
            # window.lift()
            # window.mainloop()
        except (IOError, TypeError):
            loading.pack_forget()
            self.error.pack()
            raise

    def loadRawData(self):
        """Loads data from a text file using MainWindow.loadData() and prompts the user for a slice"""
        self.error.pack_forget()
        self.newFrame.destroy()
        self.lift()
        self.newFrame = Tk.Frame(self.baseFrame)
        self.newFrame.pack(side=Tk.BOTTOM, expand=True)
        path = tkFileDialog.askopenfilename()
        extension = path[path.rfind("."):]
        print extension
        if extension != ".npy":  # in self.settings["Non Binary Extensions"]:
            instructions = Tk.Label(self.newFrame, text="The first 10 lines of your data are displayed below.")
            instructions.pack()
            lines = []
            with open(path) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        firstline = line
                    lines.append(line)
                    if i == 10:
                        break
            regex = r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?|\w+'
            try:
                numbers = re.findall(regex, firstline)
            except NameError:
                self.error.pack()
                raise
            tree = ttk.Treeview(self.newFrame, height=10)
            tree.pack()
            cols = ()
            style = ttk.Style(self)
            style.configure("Treeview", rowheight=int(.3 * self.settings["DPI"]))
            for i in range(len(numbers)):
                cols += tuple(str(i))
                # tree.heading(str(i), text="Column")
            tree["columns"] = cols
            for i in range(len(numbers)):
                tree.heading(str(i), text="Column %d" % i)
            for i, l in enumerate(lines):
                tree.insert("", "end", text="Line %d" % i, values=re.findall(regex, l))
            xFrame = Tk.Frame(self.newFrame)
            xFrame.pack(expand=True)
            xLabel = Tk.Label(xFrame, text="X-Data Column: ")
            xLabel.pack(side=Tk.LEFT)
            xEntry = Tk.Entry(xFrame, width=2)
            xEntry.insert(0, "0")
            xEntry.pack(side=Tk.RIGHT)
            yFrame = Tk.Frame(self.newFrame)
            yFrame.pack(expand=True)
            yLabel = Tk.Label(yFrame, text="Y-Data Column: ")
            yLabel.pack(side=Tk.LEFT)
            yEntry = Tk.Entry(yFrame, width=2)
            yEntry.insert(0, "1")
            yEntry.pack(side=Tk.RIGHT)
            headerVal = Tk.IntVar()
            headerVal.set(0)
            Tk.Checkbutton(self.newFrame, text="Data Contains Column Headers", variable=headerVal).pack()
            cleanVal = Tk.IntVar()
            cleanVal.set(1)
            Tk.Checkbutton(self.newFrame, text="Clean infs and NaNs (recommended)", variable=cleanVal).pack()
            chunkVal = Tk.IntVar()
            chunkVal.set(1)
            Tk.Checkbutton(self.newFrame, text="Read data in chunks (recommended)", variable=chunkVal).pack()
            Tk.Button(self.newFrame, text="Load", command=lambda: self.load(
                path, xEntry.get(), yEntry.get(), headerVal.get(), cleanVal.get(), chunkVal.get())).pack()
        else:
            self.load(path, shouldChunk=False)
        self.update()
        self.lift()

    def load(self, path, xCol=0, yCol=1, hasHeaders=False, shouldClean=True, shouldChunk=True):
        self.newFrame.destroy()
        self.newFrame = Tk.Frame(self.baseFrame)
        self.newFrame.pack(side=Tk.BOTTOM)
        self.lift()
        try:
            xCol = int(xCol)
            yCol = int(yCol)
        except ValueError:
            self.error.pack()
            raise
        loading = Tk.Label(self.newFrame, text="Loading data...")
        loading.pack()
        progress = None
        if shouldChunk:
            progress = ttk.Progressbar(self.newFrame, length=self.defaultWidth * .5, mode="indeterminate", maximum=10)
            progress.pack()
        self.update()
        try:
            data = MainWindow.loadData(path, chunkSize=self.settings['Load Chunk Size'], tkProgress=progress,
                                       tkRoot=self, xCol=xCol, yCol=yCol, header=hasHeaders, clean=shouldClean,
                                       chunkRead=shouldChunk)
        except (ValueError, IOError):
            loading.pack_forget()
            if progress: progress.pack_forget()
            self.error.pack()
            raise
        loading.pack_forget()
        if progress: progress.pack_forget()
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
        Tk.Button(self.newFrame, text="Create Project" if not self.win else "Add Graph",
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
            newDat = data[0][int(begin):int(end)], data[1][int(begin):int(end)]
        # By x value
        else:  # elif tkVar.get() == 1:
            newBegin, newEnd = np.searchsorted(data[0], np.array([np.float64(begin), np.float64(end)]))
            newDat = data[0][newBegin:newEnd], data[1][newBegin:newEnd]
        self.quit()
        self.destroy()
        if self.win:
            gr = Graph(window=self.win, title="Raw Data")
            gr.setRawData(newDat)
            self.win.addGraph(gr)
            self.win.plotGraphs()
        else:
            win = MainWindow()
            gr = Graph(window=win, title="Raw Data")
            gr.setRawData(newDat)
            win.setGraphs([[gr]])
            win.plotGraphs()
            win.mainloop()


if __name__ == "__main__":
    initial = InitialWindow()
    initial.mainloop()
