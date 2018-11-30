#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:53:42 2018

Allows print both in console and to file, also controls verbose of print function
by turning it on or of

@author: asceta
"""


import sys
import os

class print_manager(object):
    def __init__(self):
        self.original_print = print
        self.original_stdout = sys.stdout

    def config(self, file_path, log_file, verbose=True):
        file = open(os.path.join(file_path, log_file), 'w')
        self.file_printing(file)
        return self.verbose_printing(verbose)
    
    def verbose_printing(self, verbose=True):
        return self.original_print if verbose else lambda *a, **k: None
    
    def file_printing(self, file):
        sys.stdout = Output_manager(sys.stdout, file)
    
    def dont_print_in_file(self):
        sys.stdout = self.original_stdout


class Output_manager(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

if __name__ == "__main__":
    f = open('/home/asceta/Alerce/AlerceDHtest/examples/Supernovae/Refactor/out.txt', 'w')
    print_manager = print_manager()
    print_manager.file_printing(f)
    print("test", flush=True)  # This will go to stdout and the file out.txt
    print("number test %i nice" % 5, flush=True)
    print = print_manager.verbose_printing(False)
    print("This won't be printed at all", flush=True)
    print = print_manager.verbose_printing()
    print("Verbose works", flush=True)
    #use the original
    print_manager.dont_print_in_file()
    print("This won't appear on file", flush=True)  # Only on stdout
    f.close()
