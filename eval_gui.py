#   eval_gui.py
#   Using python 3.5
import copy
import tkinter as tk
from tkinter import ttk

import sys

from evaluator import main, CONFIGS, parse_config


def findfile(title):
    return tk.filedialog.askopenfilename(initialdir='./', title=title)


def fetch_config(lang_set=None, ngram=None, tokenize=None, header=None,
                 verbose=None, lang1_train=None, lang2_train=None, class_jar=None,
                 lang1_class=None, lang2_class=None, gold_path=None, gold_delimiter=None,
                 lang1_other=None, lang2_other=None, ne_tag=None, other_tags=None,
                 ner_chunk_size=None):
    edited_config = copy.deepcopy(CONFIGS)
    if lang_set is not None:
        edited_config["lang_set"] = lang_set
    if ngram is not None:
        edited_config["ngram"] = ngram
    if tokenize is not None:
        edited_config["tokenize"] = tokenize
    if header is not None:
        edited_config["header"] = header
    if verbose is not None:
        edited_config["verbose"] = verbose
    if lang1_train is not None:
        edited_config["lang1_train"] = lang1_train
    if lang2_train is not None:
        edited_config["lang2_train"] = lang2_train
    if class_jar is not None:
        edited_config["class_jar"] = class_jar
    if lang1_class is not None:
        edited_config["lang1_class"] = lang1_class
    if lang2_class is not None:
        edited_config["lang2_class"] = lang2_class
    if gold_path is not None:
        edited_config["gold_path"] = gold_path
    if gold_delimiter is not None:
        edited_config["gold_delimiter"] = gold_delimiter
    if lang1_other is not None:
        edited_config["lang1_other"] = lang1_other
    if lang2_other is not None:
        edited_config["lang2_other"] = lang2_other
    if ne_tag is not None:
        edited_config["ne_tag"] = ne_tag
    if other_tags is not None:
        edited_config["other_tags"] = other_tags
    if ner_chunk_size is not None:
        edited_config["ner_chunk_size"] = ner_chunk_size
    return edited_config


class Redirector(object):
    def __init__(self, textarea):
        self.textarea = textarea

    def write(self, msg):
        self.textarea.insert(tk.END, msg)


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
        # self.base.resizable(height=False, width=False)
        self.base.title("Bilingual Annotation Tasks Force - Python Tagger")
        # The frame
        self.pack(anchor=tk.CENTER)

    def _create_variables(self):
        if CONFIGS is None or len(CONFIGS) == 0:
            parse_config()
        # Paths
        self.gold_path = tk.StringVar(value=CONFIGS["gold_path"])
        self.lang1_train_path = tk.StringVar(value=CONFIGS["lang1_train"])
        self.lang2_train_path = tk.StringVar(value=CONFIGS["lang2_train"])

    def _create_widgets(self):
        # Gold standard data
        self.gold_path_label = ttk.Label(text="Gold Standard:")
        self.gold_path_entry = ttk.Entry(textvariable=self.gold_path)
        self.gold_path_filebutton = ttk.Button(text="...", command=self.findgoldfile)
        # Training data for language 1
        self.lang1_train_path_label = ttk.Label(text="Language 1 Training Data:")
        self.lang1_train_path_entry = ttk.Entry(textvariable=self.lang1_train_path)
        self.lang1_train_path_filebutton = ttk.Button(text="...", command=self.findlang1trainfile)
        # Training data for language 2
        self.lang2_train_path_label = ttk.Label(text="Language 2 Training Data:")
        self.lang2_train_path_entry = ttk.Entry(textvariable=self.lang2_train_path)
        self.lang2_train_path_filebutton = ttk.Button(text="...", command=self.findlang2trainfile)
        # Examination
        self.examine_button = ttk.Button(text="Examine!", command=self.launch_main)
        # Redirected ouput
        self.output_frame = ttk.Frame()
        self.output = tk.Text(master=self.output_frame)
        self.output_scroll = ttk.Scrollbar(self.output_frame)
        self.output.configure(yscrollcommand=self.output_scroll.set)
        self.output_scroll.configure(command=self.output.yview)

        self.redirected = Redirector(self.output)

    def _set_widgets(self):
        # Labels
        self.gold_path_label.grid(in_=self, row=0, column=0, sticky=tk.W, padx=10)
        self.lang1_train_path_label.grid(in_=self, row=1, column=0, sticky=tk.W, padx=10)
        self.lang2_train_path_label.grid(in_=self, row=2, column=0, sticky=tk.W, padx=10)
        # Entries and button
        self.gold_path_entry.grid(in_=self, row=0, column=1, columnspan=2, padx=10, pady=5)
        self.lang1_train_path_entry.grid(in_=self, row=1, column=1, columnspan=2, padx=10, pady=5)
        self.lang2_train_path_entry.grid(in_=self, row=2, column=1, columnspan=2, padx=10, pady=5)
        self.gold_path_filebutton.grid(in_=self, row=0, column=3, padx=10, pady=5)
        self.lang1_train_path_filebutton.grid(in_=self, row=1, column=3, padx=10, pady=5)
        self.lang2_train_path_filebutton.grid(in_=self, row=2, column=3, padx=10, pady=5)
        # Last button
        self.examine_button.grid(in_=self, row=3, column=2, columnspan=3, pady=5)
        # Output
        self.output_frame.grid(in_=self, row=4, column=0, columnspan=6, padx=10, pady=5)
        self.output_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.output.pack()
        print("Hi")
        sys.stdout = self.redirected
        sys.stderr = self.redirected
        print("Hi")

    def _set_qol(self):
        # TODO Quality of life additions
        pass

    def findgoldfile(self):
        self.gold_path.set(findfile("Gold Standard"))

    def findlang1trainfile(self):
        self.lang1_train_path.set(findfile("Training Data for Language 1"))

    def findlang2trainfile(self):
        self.lang2_train_path.set(findfile("Training Data for Language 2"))

    def launch_main(self):
        self.disable()
        config = fetch_config(gold_path=self.gold_path.get(),
                              lang1_train=self.lang1_train_path.get(),
                              lang2_train=self.lang2_train_path.get())
        main(local_config=config)
        self.enable()

    def save_config(self):
        # TODO Make it easy to save the current configuration to a file
        pass

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
