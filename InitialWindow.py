import Tkinter as Tk
import tkFileDialog
from Graph import Graph
from MainWindow import MainWindow
import tkFileDialog
import numpy as np

class InitialWindow(Tk.Tk):

    def __init__(self, *args, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        Tk.Tk.__init__(self, *args, **kwargs)
        self.wm_title("Data Manipulation")
        self.defaultWidth, self.defaultHeight = self.winfo_screenwidth() * .5, self.winfo_screenheight() * .5
        self.geometry("%dx%d+0+0" % (self.defaultWidth, self.defaultHeight))
        self.baseFrame = Tk.Frame(master=self)
        self.baseFrame.pack()
        text = Tk.Label(master=self.baseFrame, text="Test Window!")
        text.pack()
        loadbutton = Tk.Button(text="Load Project", command=self.loadProject)
        loadbutton.pack()

    @staticmethod
    def loadProject():
        path = tkFileDialog.askopenfilename()
        b = np.load(path)
        rawData, metaData = b["arr_0"], b["arr_1"]
        print metaData
        graphs = []
        for i in range(len(rawData)):
            graphs.append([])
            for j in range(len(rawData[i])):
                gr = Graph()
                gr.setRawData(rawData[i][j])
                for att in metaData[i][j]:
                    setattr(gr, att, metaData[i][j][att])
                graphs[i].append(gr)
        window = MainWindow(graphs=graphs)
        window.mainloop()



    def loadRawData(self):
        pass

    def createBlankProject(self):
        pass

if __name__ == "__main__":
    initial = InitialWindow()
    initial.mainloop()
