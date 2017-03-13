#   eval_gui.py
#   Using python 3.5

import tkinter as tk
from tkinter import ttk
from evaluator import examine


class PyTaggerApp(ttk.Frame):
    def __init__(self, base, master=None):
        super().__init__(master)
        self._prime_base(base)
        self._create_variables()
        self._create_widgets()
        self._set_widgets()
        self._set_qol()

    def _prime_base(self, base):
        # Actual window
        self.base = base
        self.base.title("Bilingual Annotation Tasks Force - Python Tagger")
        self.base.geometry('{}x{}'.format(500, 500))
        # The frame
        self.grid(row=0, column=0, sticky=(tk.N, tk.E, tk.S, tk.W))
        self.rowconfigure(index=0, weight=1)
        self.columnconfigure(index=0, weight=1)

    def _create_variables(self):
        self.gold_path = tk.StringVar()
        self.test_path = tk.StringVar()

    def _create_widgets(self):
        # Gold standard data
        self.gold_path_label = ttk.Label(text="Gold Standard:")
        self.gold_path_entry = ttk.Entry(textvariable=self.gold_path)
        self.gold_path_filebutton = ttk.Button(text="...", command=self.findgoldfile)
        # Test corpus data
        self.test_path_label = ttk.Label(text="Test Corpus:")
        self.test_path_entry = ttk.Entry(textvariable=self.test_path)
        self.test_path_filebutton = ttk.Button(text="...", command=self.findtestfile)
        # Examination
        self.examine_button = ttk.Button(text="Examine!", command=self.launch_examine)

    def _set_widgets(self):
        # Labels
        self.gold_path_label.grid(row=0, column=0, sticky=tk.W)
        self.test_path_label.grid(row=1, column=0, sticky=tk.W)
        # Entries and button
        self.gold_path_entry.grid(row=0, column=1, columnspan=2)
        self.gold_path_filebutton.grid(row=0, column=3)
        self.test_path_entry.grid(row=1, column=1, columnspan=2)
        self.test_path_filebutton.grid(row=1, column=3)
        # Last button
        self.examine_button.grid(row=2, column=2, columnspan=3)

    def _set_qol(self):
        pass

    def findgoldfile(self):
        self.gold_path.set(self.findfile("Gold Standard"))

    def findtestfile(self):
        self.test_path.set(self.findfile("Test Standard"))

    def findfile(self, title):
        return tk.filedialog.askopenfilename(initialdir='./', title=title)

    def launch_examine(self):
        args = [self.gold_path.get(), self.test_path.get()]  # gold_standard test_corpus
        self.disable()
        examine(args)
        self.enable()

    def disable(self):
        for child in self.winfo_children():
            child.configure(state='disable')

    def enable(self):
        for child in self.winfo_children():
            child.configure(state='enable')


def launch_gui():
    root_win = tk.Tk()
    app = PyTaggerApp(root_win)
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
