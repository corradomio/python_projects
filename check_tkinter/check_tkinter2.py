import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

root = tk.Tk()
root.title("Interactive Matplotlib Plot in Tkinter")

fig = Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)  # Add a subplot to the figure

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Example: Plotting a simple line
x = [i for i in range(10)]
y = [i ** 2 for i in range(10)]
line, = ax.plot(x, y)  # Store the line object for potential updates


# Example of interactive update (e.g., via a button click)
def update_plot():
    new_y = [val + 1 for val in y]  # Modify data
    line.set_ydata(new_y)  # Update the line data
    canvas.draw()  # Redraw the canvas


update_button = tk.Button(master=root, text="Update Plot", command=update_plot)
update_button.pack()

root.mainloop()
