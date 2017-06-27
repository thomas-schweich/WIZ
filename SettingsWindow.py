import json
import Tkinter as Tk


def _getValFromStringVar(stringvar):
    string_value = stringvar.get()
    try:
        return int(string_value)
    except ValueError as v:
        pass
    try:
        return float(string_value)
    except ValueError as v:
        pass
    return string_value


class SettingsWindow(Tk.Frame):
    """ Window to display and alter settings """

    private_settings = {'Icon Location'}

    def __init__(self, parent, *args, **kwargs):
        Tk.Frame.__init__(self, parent, *args, **kwargs)
        self.window = Tk.Toplevel(self)
        self.settings = {}
        with open('programSettings.json', 'r') as f:
            self.settings = json.load(f)
        self.private = {k: v for k, v in self.settings.iteritems() if k in SettingsWindow.private_settings}
        self.settings = {k: v for k, v in self.settings.iteritems() if k not in SettingsWindow.private_settings}
        self.modified_settings = {}

    def open(self):
        for k, v in sorted(self.settings.iteritems(), reverse=True):
            frame = Tk.Frame(self.window)
            Tk.Label(frame, text=k).pack(side=Tk.LEFT, fill=Tk.X)
            newVal = Tk.StringVar()
            newVal.set(v)
            self.modified_settings.update({k: newVal})
            Tk.Entry(frame, textvariable=newVal).pack(side=Tk.RIGHT, fill=Tk.X)
            frame.pack(side=Tk.TOP, fill=Tk.X)
        buttonFrame = Tk.Frame(self.window)
        Tk.Button(buttonFrame, text='Save', command=self.save).pack(side=Tk.LEFT, fill=Tk.X)
        Tk.Button(buttonFrame, text='Cancel', command=self.close).pack(side=Tk.LEFT, fill=Tk.X)
        buttonFrame.pack()

    def save(self):
        settings = {k: _getValFromStringVar(v) for k, v in self.modified_settings.iteritems()}
        settings.update(self.private)
        with open('programSettings.json', 'w') as f:
            json.dump(settings, f)
        self.close()

    def close(self):
        self.window.destroy()
