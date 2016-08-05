import Tkinter as Tk
import tkFileDialog
from ExpressionChain import ExpressionChain
import pickle


class TemplateCreator(Tk.Toplevel):
    __author__ = "Thomas Schweich"

    def __init__(self, *args, **kwargs):
        Tk.Toplevel.__init__(self, *args, **kwargs)
        Tk.Label(self, text="Create A Template").pack()
        self.instructionsFrame = Tk.Frame(self)
        self.instructionsFrame.pack(side=Tk.TOP)
        self.instructionsShown = True
        self.instructions = \
            Tk.Label(self.instructionsFrame, justify=Tk.LEFT,
                     text="* Each line you create below is evaluated as a user written expression.\n"
                          "* Lines which evaluate to graphs will be plotted to the screen when the template is loaded."
                          "\n* A graph containing the data you load using the template can be referenced by "
                          "typing '<ORIGINAL>' with no quotes.\n"
                          "* Expressions down the line can access names defined above them, but not below them. "
                          "Thus, operations can be chained.")
        self.instructions.pack(side=Tk.TOP)
        self.hidebutton = Tk.Label(self.instructionsFrame, text="Hide", fg="blue")
        self.hidebutton.bind("<Button-1>", self.showHide)
        self.hidebutton.pack(side=Tk.BOTTOM)
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
        self.dropdownFrame = Tk.Frame(self)
        self.dropdownFrame.pack(side=Tk.BOTTOM)
        self.dropdownVar = Tk.StringVar(self)
        self.dropdownVar.set("ORIGINAL")
        self.dropdown = Tk.OptionMenu(self.dropdownFrame, self.dropdownVar, "ORIGINAL")
        self.dropdown.pack(side=Tk.LEFT)
        self.addDropButton = Tk.Button(self.dropdownFrame, text="Insert", command=self.insert)
        self.addDropButton.pack(side=Tk.RIGHT)
        self.frames = []
        self.names = []
        self.nameVars = []
        self.expressions = []
        # Create first set of entries manually
        self.addExp()
        self.names[0].insert(0, "Graph 1")
        self.expressions[0].insert(0, "<ORIGINAL>")
        self.error = Tk.Label(self.baseframe, text="Invalid selection", fg="red")

    def showHide(self, _):
        if self.instructionsShown:
            self.instructions.pack_forget()
            self.hidebutton.configure(text="Show")
            self.instructionsShown = False
        else:
            self.instructions.pack()
            self.hidebutton.configure(text="Hide")
            self.instructionsShown = True

    def insert(self):
        try:
            self.focus_get().insert(Tk.INSERT, "<%s>" % self.dropdownVar.get())
        except TypeError:
            pass  # No Entry selected

    def addExp(self):
        """Creates new labels and entries, adding the entries to their respective lists for later usage"""
        newFrame = Tk.Frame(self.baseframe)
        newFrame.pack(side=Tk.TOP, fill=Tk.X)
        nameLabel = self.nameLabel(newFrame)
        nameLabel.pack(side=Tk.LEFT)
        nameVar = Tk.StringVar(self)
        nameVar.trace('w', self.updateOptions)
        self.nameVars.append(nameVar)
        name = Tk.Entry(newFrame, textvariable=nameVar)
        name.pack(side=Tk.LEFT)
        expLabel = self.expLabel(newFrame)
        expLabel.pack(side=Tk.LEFT)
        exp = Tk.Entry(newFrame, width=100)
        exp.pack(side=Tk.LEFT, fill=Tk.X)
        self.frames.append(newFrame)
        self.names.append(name)
        self.expressions.append(exp)

    def updateOptions(self, *args):
        self.dropdown.destroy()
        self.dropdown = Tk.OptionMenu(self.dropdownFrame, self.dropdownVar, "ORIGINAL", *[n.get() for n in self.nameVars])
        self.dropdown.pack(side=Tk.LEFT)

    def save(self):
        self.error.pack_forget()
        chain = ExpressionChain()
        for i, name in enumerate(self.names):
            chain.addExp(self.expressions[i].get(), name.get())
        path = tkFileDialog.asksaveasfilename(defaultextension=".wizt",
                                              filetypes=[("WIZ Template", ".wizt")])
        try:
            with open(path, 'w+') as f:
                pickle.dump(chain, f)
        except IOError:
            self.error.pack()
