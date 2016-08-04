import Tkinter as Tk
import tkFileDialog
from ExpressionChain import ExpressionChain
import pickle


class TemplateCreator(Tk.Toplevel):

    def __init__(self, *args, **kwargs):
        Tk.Toplevel.__init__(self, *args, **kwargs)
        Tk.Label(self, text="Create A Template\n"
                            "Each line you create below will add a graph to the screen when the template is loaded.\n"
                            "A graph containing the data you load using the template can be referenced by "
                            "typing '<ORIGINAL>' with no quotes.").pack()
        self.baseframe = Tk.Frame(self)
        self.baseframe.pack(fill=Tk.BOTH, side=Tk.TOP)
        # Convenience functions for creating a label that says "Name: " or "Expression: "
        self.nameLabel = lambda frame: Tk.Label(frame, text="Name: ")
        self.expLabel = lambda frame: Tk.Label(frame, text="Expression: ")
        # Create buttons in separate frame which stays on the bottom of the window
        self.buttonFrame = Tk.Frame(self)
        self.buttonFrame.pack(side=Tk.BOTTOM)
        self.addButton = Tk.Button(self.buttonFrame, text="Add Expression", command=self.addExp)
        self.addButton.pack(side=Tk.LEFT)
        self.saveButton = Tk.Button(self.buttonFrame, text="Save Template", command=self.save)
        self.saveButton.pack(side=Tk.RIGHT)
        self.frames = []
        self.names = []
        self.expressions = []
        # Create first set of entries manually
        self.addExp()

    def addExp(self):
        """Creates new labels and entries, adding the entries to their respective lists for later usage"""
        newFrame = Tk.Frame(self.baseframe)
        newFrame.pack(side=Tk.TOP, fill=Tk.X)
        nameLabel = self.nameLabel(newFrame)
        nameLabel.pack(side=Tk.LEFT)
        name = Tk.Entry(newFrame)
        name.pack(side=Tk.LEFT)
        expLabel = self.expLabel(newFrame)
        expLabel.pack(side=Tk.LEFT)
        exp = Tk.Entry(newFrame, width=100)
        exp.pack(side=Tk.LEFT, fill=Tk.X)
        self.frames.append(newFrame)
        self.names.append(name)
        self.expressions.append(exp)

    def save(self):
        chain = ExpressionChain()
        for i, name in enumerate(self.names):
            chain.addExp(self.expressions[i].get(), name.get())
        path = tkFileDialog.asksaveasfilename(defaultextension=".gee",
                                              filetypes=[("WIZ Template", ".gee")])
        with open(path, 'w+') as f:
            pickle.dump(chain, f)
