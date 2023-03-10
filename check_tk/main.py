# -*- coding: utf-8 -*-
import tkinter as tk

win = tk.Tk()

w = 600 # Width
h = 300 # Height

screen_width = win.winfo_screenwidth()  # Width of the screen
screen_height = win.winfo_screenheight() # Height of the screen

# Calculate Starting X and Y coordinates for Window
x = (screen_width//2) - (w//2)
y = (screen_height//2) - (h//2)

win.geometry(f"{w}x{h}+{x}+{y}")

win.mainloop()
