import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import json
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config_GUI import api_gui, make_zstack_simulation, fit_aberrations, fit_emitters
import re
import numpy as np
from tkinter import messagebox, simpledialog
from PIL import ImageTk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter import Tk, filedialog, messagebox
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tifffile

import tifffile
from matplotlib.widgets import Slider
params_dict = {}
root = tk.Tk()

def handle_exceptions(func):
    import traceback
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred:\n\n{str(e)}\n\n"
            trace = traceback.format_exc()
            error_message += f"File and line of code:\n{trace}\n\nPlease try to fix it or open an issue at:\nhttps://github.com/pvanvelde/VectorialPSF"
            show_error_popup("Error", error_message)
    return wrapper
@handle_exceptions
def view_psf():

    def save_psf():
        # Function to save the PSF as a TIFF file
        root.config(cursor="watch")
        root.update()
        # Ask the user to select a directory and filename for saving the file
        save_path = filedialog.asksaveasfilename(defaultextension=".tiff",
                                                 filetypes=[("TIFF files", "*.tiff"), ("All Files", "*.*")])

        if save_path:
            # Save the PSF data as a TIFF file
            tifffile.imwrite(save_path, psf_data, imagej=True)
            save_label.config(text=f"PSF saved as {save_path}")
        else:
            save_label.config(text="No file selected. PSF not saved.")
        root.config(cursor="")
        root.update()

    def close_window():
        # Function to close the PSF viewer window and the figure
        plt.close(fig)
        psf_window.destroy()

    def save_graphs(fig):
        # Function to save the graphs to a file
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if file_path:
            if not file_path.endswith(".png"):
                file_path += ".png"
            fig.savefig(file_path)
            messagebox.showinfo("Save Graphs", "Graphs saved successfully.")

    def save_numbers(data, z_values, names):
        # Function to save the numbers as a table in a text file
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt")])
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += ".txt"
        if file_path:
            num_cols = len(z_values) + 1  # Number of columns in the table (z values + header column)
            num_rows = len(names) + 1  # Number of rows in the table (names + data rows)

            table_data = np.zeros((num_rows, num_cols), dtype=object)  # Initialize the table data

            # Reshape the z_values array to have the correct shape
            z_values = z_values.reshape(-1)

            # Fill the header row with z values
            table_data[0, 0] = 'z [nm]'
            table_data[0, 1:] = z_values

            # Fill the first column with names
            table_data[1:, 0] = names

            # Transpose the data array before assigning it to table_data
            table_data[1:, 1:] = data.T

            # Save the table to the text file
            with open(file_path, 'w') as f:
                for row in table_data:
                    f.write('\t'.join(str(value) for value in row) + '\n')

            messagebox.showinfo("Save Numbers", "Numbers saved successfully.")

    def display_plots():
        # Function to display plots based on input parameters
        # Prompt the user for photon signal and background signal
        photon_signal = simpledialog.askfloat("Enter Photon Signal PSF", "Please enter the signal of PSF (photons):",
                                              initialvalue=dialog.photon_signal)
        background_signal = simpledialog.askfloat("Enter Background Signal",
                                                  "Please enter the background signal (photons/pixel):",
                                                  initialvalue=dialog.background_signal)

        _, _, _, _, _, _, crlbdis, z_plot = make_zstack_simulation(params_dict, photon_signal,
                                                                   background_signal,False, delta_z)  # Replace `your_function` with the actual function you want to call

        # Create a new window for the plots
        plot_window = tk.Toplevel(psf_window)
        plot_window.geometry("800x800")
        plot_window.title("Plots - Cramer Rao Lower Bound (CRLB)")
        crlbdis = crlbdis.detach().cpu().numpy()

        # Create the plots using the provided values
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 14))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)


        names = ['CRLB in y [nm]', 'CRLB in x [nm]', 'CRLB in z [nm]', 'CRLB in signal [photons]', 'CRLB in background [photons/px]']

        for i, ax in enumerate(axes.flat):
            if i < 5:
                # Generate data for each subplot (replace with your actual data generation logic)
                x = z_plot
                y = crlbdis[:, i]

                ax.plot(x, y)
                ax.set_xlabel('z (nm)')
                ax.set_ylabel(names[i])

        plt.tight_layout(pad=7)

        # Add menu options for saving graphs and numbers
        plot_menu = tk.Menu(plot_window)
        plot_window.config(menu=plot_menu)

        file_menu = tk.Menu(plot_menu)
        plot_menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Graphs", command=lambda: save_graphs(fig))
        file_menu.add_command(label="Save Numbers", command=lambda: save_numbers(crlbdis, z_plot, names))

        # Set up canvas for displaying the plots
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    class InputDialog(simpledialog.Dialog):
        def body(self, master):
            self.photon_signal_label = tk.Label(master, text="Photon Signal PSF (photons):")
            self.photon_signal_entry = tk.Entry(master)
            self.background_signal_label = tk.Label(master, text="Background Signal (photons/pixel):")
            self.background_signal_entry = tk.Entry(master)
            self.delta_z_label = tk.Label(master, text="\u0394z between slices (nm):")
            self.delta_z_entry = tk.Entry(master)
            self.poisson_noise_var = tk.BooleanVar(value=False)

            self.poisson_noise_checkbutton = tk.Checkbutton(master, variable=self.poisson_noise_var,
                                                            text=" Check for poisson noise")

            self.photon_signal_label.grid(row=0, column=0, sticky=tk.W)
            self.photon_signal_entry.grid(row=0, column=1)
            self.background_signal_label.grid(row=1, column=0, sticky=tk.W)
            self.background_signal_entry.grid(row=1, column=1)
            self.delta_z_label.grid(row=2, column=0, sticky=tk.W)
            self.delta_z_entry.grid(row=2, column=1)
            self.poisson_noise_checkbutton.grid(row=3, column=0, sticky=tk.W)
            # Set default values
            self.photon_signal_entry.insert(tk.END, "4000.0")
            self.background_signal_entry.insert(tk.END, "10.0")
            self.delta_z_entry.insert(tk.END, "5")

        def apply(self):
            self.photon_signal = float(self.photon_signal_entry.get())
            self.background_signal = float(self.background_signal_entry.get())
            self.delta_z = float(self.delta_z_entry.get())
            self.poisson_noise = self.poisson_noise_var.get()

    dialog = InputDialog(root, "Input Dialog")
    photon_signal = dialog.photon_signal
    background_signal = dialog.background_signal
    delta_z = dialog.delta_z
    poisson_noise = dialog.poisson_noise

    # Create a Tkinter window for displaying the PSF
    psf_window = tk.Toplevel(root)
    psf_window.title("PSF Viewer")
    psf_window.geometry("800x700")
    save_label = ttk.Label(psf_window, text="")
    save_label.pack()

    # Load the PSF data (replace this with your actual 3D PSF data)


    psf_data, _, zmin, zmax, pupil,_, crlb, _ = make_zstack_simulation(params_dict,photon_signal,background_signal, poisson_noise, delta_z)
    psf_data = psf_data.detach().cpu().numpy()

    # Create a Figure and Axes for displaying the PSF slice
    fig, ax = plt.subplots(1,2,figsize=(6, 4))
    #fig.tight_layout(pad=1)
    ax[0].axis('off')  # Remove the axes
    ax[0].set_title('PSF')
    img_slice = ax[0].imshow(psf_data[0], cmap='viridis', vmin=0,vmax=np.amax(psf_data))

    # Create a slider for selecting the slice
    slider_frame = ttk.Frame(psf_window)
    slider_frame.pack(pady=10)

    slider_label = ttk.Label(slider_frame, text="Select Slice:")
    slider_label.pack(side=tk.LEFT)

    canvas = FigureCanvasTkAgg(fig, master=psf_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    def update_slice(event):
        # Update the displayed slice based on the slider value
        current_slice = int(slider.get())
        img_slice.set_data(psf_data[current_slice])
        canvas.draw()

    slider = ttk.Scale(slider_frame, from_=0, to=len(psf_data) - 1, orient=tk.HORIZONTAL, length=400,
                       command=update_slice)
    slider.pack(side=tk.LEFT)
    slider.set(0)  # Set the initial slice to 0

    # Create a label to display the PSF range
    psf_range_label = ttk.Label(psf_window, text=f"PSF Range: {zmin} nm to {zmax} nm")
    psf_range_label.pack(pady=10)

    # Display the 2D pupil image using imshow
     # Create a separate figure for displaying the pupil image
    pupil_data = np.real(pupil.detach().cpu().numpy())
    max_value = np.max(np.abs(pupil_data))

    # Adjust vmin and vmax based on the maximum absolute value
    vmin = -2 if max_value < 2 else None
    vmax = 2 if max_value < 2 else None
    if params_dict['z_stack'] == True:
        im = ax[1].imshow(pupil_data.mean(0), cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        im = ax[1].imshow(pupil_data, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].axis('off')  # Remove the axes
    ax[1].set_title("Pupil Function")

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Phase (nm)')
    # Ad    d a button to close the PSF viewer window
    close_button = ttk.Button(psf_window, text="Close", command=close_window)
    close_button.pack(pady=10)

    # Add a button to save the PSF as a TIFF file
    save_button = ttk.Button(psf_window, text="Save PSF as TIFF", command=save_psf)
    save_button.pack(pady=10)

    # Add a button to display the table
    table_button = ttk.Button(psf_window, text="Display CRLB plots", command=display_plots)
    table_button.pack(pady=10)



@handle_exceptions
def show_error_popup(title, message):
    from tkinter import scrolledtext
    error_popup = tk.Toplevel(root)
    error_popup.title(title)
    error_popup.geometry("400x300")

    text_widget = scrolledtext.ScrolledText(error_popup, wrap=tk.WORD, width=40, height=10)
    text_widget.insert(tk.END, message)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(padx=20, pady=20)

    def close_program():
        error_popup.destroy()
        root.destroy()

    ok_button = ttk.Button(error_popup, text="OK", command=close_program)
    ok_button.pack(pady=10)

    # Set the error popup to be on top and grab focus
    error_popup.grab_set()
    error_popup.focus_set()


@handle_exceptions
def fit_vectorial_psf():
    def show_loading_popup(root):
        popup = tk.Toplevel(root)
        popup.title("Loading...")
        popup.geometry("200x100")

        label = tk.Label(popup, text="Please wait, \nfitting emitters, \nthis may take a while...", padx=20, pady=20)
        label.pack()

        return popup
        # Ask the user to open the .tif(f) file
    roi_file_path = filedialog.askopenfilename(filetypes=[("TIFF Files", "*.tif *.tiff")],title = 'Open candidate emmitters ROI')

    if roi_file_path:
        # Load the .tif(f) file as a NumPy array
        roi_array = tifffile.imread(roi_file_path)

        # Ask the user to open the .txt file
        roi_pos_file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")],title = 'Open ROI pos [frame, row, collumn]')

        if roi_pos_file_path:
            # Load the .txt file as a NumPy array
            roi_pos_array = np.loadtxt(roi_pos_file_path)

            save_path_hdf5 = filedialog.asksaveasfilename(defaultextension=".hdf5",
                                                          filetypes=[("HDF5 Files", "*.hdf5")],
                                                          title="Save results as .hdf5")

        else:
            messagebox.showwarning("File Not Selected", "No ROI position file selected. Aberration fitting aborted.")
    else:
        messagebox.showwarning("File Not Selected", "No ROI file selected. Aberration fitting aborted.")

    # Perform the fitting and get 'mushow' and 'spotsshow'
    popup = show_loading_popup(root)
    root.update()
    mushow, spotsshow,s = fit_emitters(params_dict, roi_array, roi_pos_array, save_path_hdf5)
    popup.destroy()
    outcome_stacked = np.concatenate((spotsshow,mushow),axis=-1)
    # Create the main window
    fit_abber_window = tk.Toplevel(root)
    fit_abber_window.title("PSF Viewer")
    fit_abber_window.geometry("1000x700")

    # Create a Canvas widget for the main window
    canvas = tk.Canvas(fit_abber_window)
    canvas.pack(side="left", fill="both", expand=True)
    content = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=content, anchor="nw")

    # # Create a Label widget to display the string s1
    # s1_label = ttk.Label(content, text=s, wraplength=600, justify=tk.LEFT)
    # s1_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    # Create a Scrollbar for the Canvas
    scrollbar = ttk.Scrollbar(fit_abber_window, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    # Configure the Canvas to use the Scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a Frame inside the Canvas to hold the content
    content = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=content, anchor="nw")

    # Create figure and axes for the initial slice
    fig, ax1= plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    plt.subplots_adjust(left=0.1, bottom=0.25)

    current_slice = 0
    img = ax1.imshow(outcome_stacked[current_slice, :, :], cmap='viridis')
    ax1.set_title(f"Random set of emitters, slice {current_slice + 1}")

    # Create a slider for selecting the slice
    slider_frame = ttk.Frame(fit_abber_window)
    slider_frame.pack(pady=10)

    # Update the image and title based on the selected slice
    def update_slice(val):
        current_slice = int(slider.get())
        img.set_data(outcome_stacked[current_slice, :, :])
        img.set_clim(vmin=np.min(outcome_stacked), vmax=np.max(outcome_stacked))

        ax1.set_title(f"Slice {current_slice + 1}")
        fig.canvas.draw_idle()

    slider_label = ttk.Label(slider_frame, text="Select Slice:")
    slider_label.pack(side=tk.LEFT)
    slider = ttk.Scale(slider_frame, from_=0, to=np.shape(outcome_stacked)[0] - 1, orient=tk.HORIZONTAL,
                       length=400, command=update_slice)
    slider.pack(side=tk.LEFT)
    slider.set(0)  # Set the initial slice to 0

    canvas = FigureCanvasTkAgg(fig, master=fit_abber_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Create a Label widget to display the string s1 at the bottom left
    max_line_length = 30  # Maximum characters per line
    split_savefn = [save_path_hdf5[i:i + max_line_length] for i in range(0, len(save_path_hdf5), max_line_length)]
    formatted_savefn = '\n'.join(split_savefn)  # Join lines with newline characters
    content.grid_rowconfigure(1, weight=5)
    s1_label = ttk.Label(content, text=s+'Saved in '+formatted_savefn, wraplength=1500, justify=tk.LEFT)
    s1_label.grid(row=1, column=0, columnspan=4, padx=10, pady=600, sticky="sw")



@handle_exceptions
def fit_aberrations_bead_stacks():
    class AberrationDialog:
        def __init__(self, parent, val):
            self.parent = parent
            self.dialog = tk.Toplevel(parent)
            self.dialog.title("Please Wait")
            self.dialog.geometry("700x150")

            message = f"Fitting aberrations on {val} beads, this may take a while.\nPlease pick an accurate initial guess for the aberrations and click 'Continue' to proceed..."
            label = tk.Label(self.dialog, text=message, padx=20, pady=20)
            label.pack()

            continue_button = ttk.Button(self.dialog, text="Continue", command=self.on_continue)
            continue_button.pack(pady=10)

            self.dialog.transient(parent)
            self.dialog.grab_set()
            parent.wait_window(self.dialog)

        def on_continue(self):
            self.dialog.destroy()
            choose_own_aberrations(entries["Number of Zernike aberrations to fit/use \n (start from Noll index=5) [int]"].get())

    def show_loading_popup(root):
        popup = tk.Toplevel(root)
        popup.title("Loading...")
        popup.geometry("200x100")

        label = tk.Label(popup, text="Please wait...", padx=20, pady=20)
        label.pack()



        return popup
    # Start off with the Zernikes up to j=15
    _noll_n = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    _noll_m = [0, 0, 1, -1, 0, -2, 2, -1, 1, -3, 3, 0, 2, -2, 4, -4]

    def noll_to_zern(j):
        """Convert linear Noll index to tuple of Zernike indices.
        j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike
        index.

        c.f. https://oeis.org/A176988

        Parameters:
            j:      Zernike mode Noll index

        Returns:
            (n, m) tuple of Zernike indices
        """
        while len(_noll_n) <= j:
            n = _noll_n[-1] + 1
            _noll_n.extend([n] * (n + 1))
            if n % 2 == 0:
                _noll_m.append(0)
                m = 2
            else:
                m = 1
            # pm = +1 if m values go + then - in pairs.
            # pm = -1 if m values go - then + in pairs.
            pm = +1 if (n // 2) % 2 == 0 else -1
            while m <= n:
                _noll_m.extend([pm * m, -pm * m])
                m += 2

        return _noll_n[j], _noll_m[j]
    # Request to load a .tif file
    filepath = filedialog.askopenfilename(title="Open bead stack, format: (numbeads, z-slices, rows, collumns)", filetypes=[("TIFF Files", "*.tif *.tiff")])

    if filepath:
        beads = tifffile.imread(filepath)
        #print(np.shape(beads))
        if beads.ndim == 3:
            beads = beads[None, ...]
        # Create a Tkinter window for displaying the PSF



        # messagebox.showinfo("Please Wait",
        #                     f"Fitting aberrations on {np.shape(beads)[0]} beads, this may take a while. \n Please pick a accurate intial guess for the aberrations and press Ok to continue...")

        aberration_dialog = AberrationDialog(root, np.shape(beads)[0])

        # message = f"Fitting aberrations on {np.shape(beads)[0]} beads, this may take a while. \n Please pick an accurate initial guess for the aberrations and click 'Continue' to proceed..."
        # messagebox.showinfo("Please Wait", message)
        #
        # # Create a button to continue
        # continue_button = ttk.Button(root, text="Continue", command=lambda:  choose_own_aberrations(entries["Number of Zernike aberrations to fit/use \n (start from Noll index=5) [int]"].get()))
        # # continue_button.grid(pady=10)
        # root.update()
        # Display loading popup


        loading_popup = show_loading_popup(root)
        root.update()  # Update the GUI to show the loading popup
        outcome, theta, traces = fit_aberrations(params_dict, beads)
        loading_popup.destroy()
        fit_abber_window = tk.Toplevel(root)
        fit_abber_window.title("PSF Viewer")
        fit_abber_window.geometry("1000x700")
        fit_abber_window.lift()
        save_label = ttk.Label(fit_abber_window, text="")
        save_label.pack()
        # Close the loading popup
        #loading_popup.destroy()
        messagebox.showinfo("Process Completed", "Fitting aberrations completed.")

        outcome = np.transpose(outcome, (0, 3, 1, 2))
        outcome_stacked = np.reshape(outcome, (-1, np.shape(outcome)[-2], np.shape(outcome)[-1]))
        # outcome_sq = np.squeeze(outcome, axis=0)
        # outcome_stacked = np.reshape(outcome_sq, (-1,) + outcome_sq.shape[1:])
        #beads_sq = np.squeeze(beads, axis=0)
        beads_stacked = np.reshape(beads, (-1, np.shape(beads)[-2], np.shape(beads)[-1]))
        outcome_stacked = np.concatenate((beads_stacked, outcome_stacked), axis=-1)

        # Show image slices of the 'outcome' matrix
        zslices, ROIsize, ROIsize = outcome_stacked.shape

        # Create figure and axes for the initial slice
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        plt.subplots_adjust(left=0.1, bottom=0.25)

        current_slice = 0
        img = ax1.imshow(outcome_stacked[current_slice, :, :], cmap='viridis')
        ax1.set_title(f"Slice {current_slice + 1}")

        # Create a slider for selecting the slice
        slider_frame = ttk.Frame(fit_abber_window)
        slider_frame.pack(pady=10)
        # Update the image and title based on the selected slice
        def update_slice(val):
            current_slice = int(slider.get())
            img.set_data(outcome_stacked[current_slice, :, :])
            img.set_clim(vmin=np.min(outcome_stacked), vmax=np.max(outcome_stacked))

            ax1.set_title(f"Slice {current_slice + 1}")
            fig.canvas.draw_idle()
        slider_label = ttk.Label(slider_frame, text="Select Slice:")
        slider_label.pack(side=tk.LEFT)
        slider = ttk.Scale(slider_frame, from_=0, to=np.shape(outcome_stacked)[0] - 1, orient=tk.HORIZONTAL,
                           length=400, command=update_slice)
        slider.pack(side=tk.LEFT)
        slider.set(0)  # Set the initial slice to 0

        # Set common x and y labels
        # fig.text(0.5, 0.02, 'X', ha='center')
        # fig.text(0.08, 0.5, 'Y', va='center', rotation='vertical')
        theta = theta[:, 5::]
        # Show line graph with standard deviation
        mean_theta = np.mean(theta, axis=0)
        std_theta = np.std(theta, axis=0)

        # Show line graph with error bars
        ax2.errorbar(range(mean_theta.shape[0]), mean_theta, yerr=std_theta, fmt='o', label='Mean with Error Bars')

        # Set x-axis tick values and labels
        tick_values = range(mean_theta.shape[0])  # Custom tick values

        tick_labels = ['{}'.format(noll_to_zern(i)) for i in tick_values]  # Custom tick labels
        ax2.set_xticks(tick_values)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')

        # Set y-axis label and title
        ax2.set_ylabel('RMS [nm]')
        #ax2.set_title('Line Graph with Error Bars')
        ax2.set_xlabel('Zernike [Noll index]')
        # Remove the line connecting the markers
        ax2.plot([], [])  # Dummy plot to show only markers and error bars

        # Create a FigureCanvasTkAgg instance and pack it into the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=fit_abber_window)
        canvas.draw()
        canvas.get_tk_widget().pack()



        def save_psf():
            # Save the PSF matrix from the slider
            save_filepath = filedialog.asksaveasfilename(defaultextension='.tif')
            if save_filepath:
                tifffile.save(save_filepath, outcome)

        def save_aberrations():
            # Ask the user to choose a file path and extension
            theta_filepath = filedialog.asksaveasfilename(
                title='Save as .txt or .npy',
                defaultextension='.txt',
                filetypes=[('Text files', '*.txt'), ('NumPy files', '*.npy')]
            )

            if theta_filepath:
                # Determine the chosen extension
                file_extension = theta_filepath.split('.')[-1]

                if file_extension == 'txt':
                    # Save the theta as a text file in table format
                    np.savetxt(theta_filepath, theta, fmt='%.6f', delimiter='\t')
                    info_message = (
                        "Aberrations saved as [numbeads, abberations] as a text file.\n"
                        "These are the abberations for each individual bead."
                    )
                    messagebox.showinfo("Save Successful", info_message)
                elif file_extension == 'npy':
                    # Save the mean of theta as a NumPy binary file
                    abberation_arr = np.zeros((int(len(mean_theta)), 3))
                    for i in range(len(mean_theta)):
                        n, m = noll_to_zern(i + 5)
                        abberation_arr[i, 0] = n
                        abberation_arr[i, 1] = m
                        abberation_arr[i, 2] = float(mean_theta[i])
                    np.save(theta_filepath, abberation_arr)
                    info_message = (
                        "Aberrations saved as a NumPy binary file.\n"
                        "This is the mean of the abberations for all beads."
                    )
                    messagebox.showinfo("Save Successful", info_message)
                else:
                    messagebox.showwarning("Invalid Extension",
                                           "Invalid file extension selected. Aberrations not saved.")

        # Create buttons for saving PSF and aberrations
        save_psf_button = ttk.Button(fit_abber_window, text="Save PSF", command=save_psf)
        save_psf_button.pack()

        save_aberrations_button = ttk.Button(fit_abber_window, text="Save Aberrations: \n .npy: for later use in this GUI \n .txt: for own use", command=save_aberrations)
        save_aberrations_button.pack()
        fit_abber_window.mainloop()

        messagebox.showinfo("Process Completed", "File loaded!")
    else:
        messagebox.showwarning("File Not Selected", "No file selected. Aberration fitting aborted.")



@handle_exceptions
def save_outcome_tiff(outcome):
    # Function to save the outcome as a TIFF file
    filepath = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF Files", "*.tif *.tiff")])

    if filepath:
        tifffile.imsave(filepath, outcome)
        messagebox.showinfo("Success", "Outcome saved as TIFF file.")
@handle_exceptions
def save_theta_txt(theta):
    # Function to save the theta as a TXT file
    filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])

    if filepath:
        np.savetxt(filepath, theta)
        messagebox.showinfo("Success", "Theta saved as TXT file.")
@handle_exceptions
def show_result_screen():
    result_window = tk.Toplevel(root)
    result_window.title("Result Screen")
    result_window.geometry("600x600")
    def close_result_screen():
        result_window.destroy()
    def load_and_fit_stack():
        close_result_screen()
        fit_aberrations_bead_stacks()
    def load_and_fit_emit():
        close_result_screen()
        fit_vectorial_psf()
    # Thanks message
    thanks_label = ttk.Label(result_window, text="Thanks for submitting your microscope parameters!",
                             font=("Helvetica", 12, "bold"))
    thanks_label.pack(pady=20)

    # Options for the user
    options_label = ttk.Label(result_window, text="With those parameters, you can choose to:",
                              font=("Helvetica", 10))
    options_label.pack(pady=10)

    # Option 1: Simulate and View PSF of Input Values
    option1_frame = ttk.LabelFrame(result_window, text="Simulate, view, and compute CRLB PSF given input values")
    option1_frame.pack(pady=20, padx=50, ipadx=10, ipady=10)

    def view_psf_and_close():
        view_psf()
        close_result_screen()
    view_psf_button = ttk.Button(option1_frame, text="View PSF", command=view_psf_and_close)
    view_psf_button.pack(pady=5)

    # Option 2: Fit Aberrations from Z-Stack
    option2_frame = ttk.LabelFrame(result_window, text="Fit Aberrations from z-stack.")
    option2_frame.pack(pady=20, padx=50, ipadx=10, ipady=10)

    mode_a_button = ttk.Button(option2_frame, text="Load through-focus image of beads (z-stack):\n"
                                                   "- File extension should be .tif(f)\n"
                                                   "- Size: [number of beads, zslices, ROIsize, ROIsize]",
                               command=load_and_fit_stack)
    mode_a_button.pack(pady=5)

    # Option 3: Load Aberrations File and Fit
    option3_frame = ttk.LabelFrame(result_window, text="Fit candidate emitters")
    option3_frame.pack(pady=20, padx=50, ipadx=10, ipady=10)

    load_file_button = ttk.Button(option3_frame, text="Fit emitters", command=load_and_fit_emit)
    load_file_button.pack(pady=5)


    # Back button
    back_button = ttk.Button(result_window, text="Back", command=result_window.destroy)
    back_button.pack(pady=10)
@handle_exceptions
def load_variables_as_params(variables):
    global params_dict
    variable_mappings = {
        "Max Iterations for Fit [int]": "Nitermax",
        "Tolerance [float]": "tolerance",
        "Device ['cuda' or 'cpu']": "dev",
        "NA [float]": "NA",
        "Refractive index medium [float]": "refmed",
        "Refractive index coverglass [float]": "refcov",
        "Refractive index immersion oil [float]": "refimm",
        "Free working distance obj. (nm) [float]": "fwd",
        "min z-Range of bead stack (nm) [float]": "zmin",
        "max z-Range of bead stack (nm) [float]": "zmax",
        "min z-spread of estimator (nm) [float]": "zspread[0]",
        "max z-spread of estimator (nm) [float]": "zspread[1]",
        "emission wavelength (nm) [float]": "Lambda",
        "Pupil size in pixels [even int]": "Npupil",
        "Pixel Size in image plane (nm) [float]": "pixelsize",
        "Roisize in pixels [even int]": "K",
        "Number of Zernike aberrations to fit/use \n (start from Noll index=5) [int]": "numaber",
        "Batch size for fitting [int] (to prevent memory issues)": "batch_size",
        "Damping factor LM algorithm [float]": "lambda_damping",
        "Width image (only relevant for SMLM) [int]": "width",
        "Height image (only relevant for SMLM) [int]": "height"
    }



    for label, value in variables.items():
        param_name = variable_mappings.get(label)
        params_dict[param_name] = value.get()

    params_dict['depth'] = entry_imaging_depth.get()
    params_dict['zstage'] = entry_z_pos.get()
    params_dict['z_stack'] = check_zstack.get()
    if bool(params_dict['z_stack']):
        params_dict['zrange[0]'] = entry_zrange_min.get()
        params_dict['zrange[1]'] = entry_zrange_max.get()
@handle_exceptions
def toggle_zstack():
    if check_zstack.get() == 1:
        entry_zrange_min.config(state=tk.NORMAL)
        entry_zrange_max.config(state=tk.NORMAL)
    else:
        entry_zrange_min.config(state=tk.DISABLED)
        entry_zrange_max.config(state=tk.DISABLED)
@handle_exceptions
def load_params_from_file():
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, "r") as file:
            loaded_params = json.load(file)

        # Update the entry fields with the loaded parameter values
        for param, value in loaded_params.items():
            if param in entries:
                entry = entries[param]
                entry.delete(0, tk.END)
                entry.insert(0, value)
@handle_exceptions
def load_aberration_file(update_dialog = False):
    global params_dict
    file_path = filedialog.askopenfilename(filetypes=[("NumPy Files", "*.npy")])
    if file_path:
        # Load the aberration file using numpy
        aberration_array = np.load(file_path)
        print(aberration_array)
        dim1 = np.size(aberration_array,0)
        dim2 = np.size(aberration_array, 1)
        # Check if the aberration array exists

        if aberration_array is not None and dim1>0 and dim2==3:
            # Add the aberration array to the parameters dictionary
            print("Aberration file loaded successfully! \n")

            # Show a message box with the success message
            messagebox.showinfo("Success", "Aberration file loaded successfully!")
            params_dict['aberrations']=aberration_array

        if update_dialog:
            return aberration_array
    else:
            messagebox.showinfo("No success...", "check file format, size must be \n [x,3], where x is the number of zernike\n"
                                                 "polynomials. Please ensure the correct indices are used as well")
    # Call the function to load variables as parameters
@handle_exceptions
def save_params_to_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    if file_path:
        # Retrieve the parameter values from the entry fields
        params = get_entry_field_values()

        with open(file_path, "w") as file:
            json.dump(params, file, indent=4)
@handle_exceptions
def submit():
    global params_dict
    # Perform validation of entry fields
    params = validate_entry_fields()
    if params is None:
        return
    load_variables_as_params(entries)

    #print(params_dict)
    if params_dict["depth"] == '':
        params_dict["depth"] = 0
        params_dict["zstage"] = 0
        response = messagebox.askquestion("Warning",
                                          "Warning: The imaging depth and stage position were not filled in and are set to 0. \n\nContinue?")
        if response == 'no':
            return
    if -float(params_dict["zspread[0]"]) > float(params_dict["depth"]):
        # Raise alarm
        response = messagebox.askquestion("Warning", "Warning: The estimated spread of z-values is greater than the imaging depth, \nSwhich may lead to unrealistic outcomes.\n\nContinue?")
        if response == 'no':
            return

    # Perform further actions with the parameters (e.g., start the fitting process)
    show_result_screen()

@handle_exceptions
def validate_entry_fields():
    params = get_entry_field_values()

    # Perform validation on the parameter values
    # Add your validation logic here

    return params

@handle_exceptions
def get_entry_field_values():
    params = {}
    for param, entry in entries.items():
        value = entry.get()
        if value:
            if is_numeric(value):
                params[param] = int(value) if value.isdigit() else float(value)
            elif value.lower() == "true":
                params[param] = True
            elif value.lower() == "false":
                params[param] = False

            else:
                params[param] = value

    return params

@handle_exceptions
def clear_entry_fields():
    for entry in entries.values():
        entry.delete(0, tk.END)
@handle_exceptions
def choose_own_aberrations(val):
    # Start off with the Zernikes up to j=15
    _noll_n = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    _noll_m = [0, 0, 1, -1, 0, -2, 2, -1, 1, -3, 3, 0, 2, -2, 4, -4]

    def load_aberration_values(entry_points):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy Files", "*.npy")])
        if file_path:
            aberration_array = np.load(file_path)
            if aberration_array.shape[0] == len(entry_points):
                for i, entry in enumerate(entry_points):
                    entry.delete(0, tk.END)
                    entry.insert(0,
                                 str(aberration_array[i, 2]))  # Assuming the aberration values are in the third column
            else:
                messagebox.showerror("Error", "Loaded aberration values do not match the number of entry points.")
        else:
            messagebox.showinfo("No File Selected", "No file was selected. Please select a valid NumPy file.")

    def use_aberration_values():
        global params_dict
        aberration_array = np.zeros((int(val), 3))
        for i in range(len(entry_points)):
            n, m = noll_to_zern(i + 5)
            aberration_array[i, 0] = n
            aberration_array[i, 1] = m
            aberration_array[i, 2] = float(entry_points[i].get())

        params_dict['aberrations'] = aberration_array
        aberration_window.destroy()

    def noll_to_zern(j):
        """Convert linear Noll index to tuple of Zernike indices.
        j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike
        index.

        c.f. https://oeis.org/A176988

        Parameters:
            j:      Zernike mode Noll index

        Returns:
            (n, m) tuple of Zernike indices
        """
        while len(_noll_n) <= j:
            n = _noll_n[-1] + 1
            _noll_n.extend([n] * (n + 1))
            if n % 2 == 0:
                _noll_m.append(0)
                m = 2
            else:
                m = 1
            # pm = +1 if m values go + then - in pairs.
            # pm = -1 if m values go - then + in pairs.
            pm = +1 if (n // 2) % 2 == 0 else -1
            while m <= n:
                _noll_m.extend([pm * m, -pm * m])
                m += 2

        return _noll_n[j], _noll_m[j]
    def save_aberration_values(values=0):
        # Retrieve the values from the entry points
        values = [entry.get() for entry in entry_points]
        abberation_arr = np.zeros((int(val), 3))
        for i in range(len(entry_points)):
            n, m = noll_to_zern(i + 5)
            abberation_arr[i, 0] = n
            abberation_arr[i, 1] = m
            abberation_arr[i, 2] = float(entry_points[i].get())

        # Validate the values or perform any necessary processing

        # Get the file path from the user
        file_path = filedialog.asksaveasfilename(defaultextension='*.npy')

        if file_path:


            # Save the array as an .npy file
            np.save(file_path, abberation_arr)

        # Close the aberration window
        aberration_window.destroy()

    # Create a new Toplevel window for aberration input
    aberration_window = tk.Toplevel(root)
    aberration_window.title("Aberration Input")
    aberration_window.geometry("500x600")
    # Create a Canvas widget for the aberration window
    canvas = tk.Canvas(aberration_window)
    canvas.pack(side="left", fill="both", expand=True)

    # Create a Scrollbar for the Canvas
    scrollbar = ttk.Scrollbar(aberration_window, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    # Configure the Canvas to use the Scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a Frame inside the Canvas to hold the content
    content = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=content, anchor="nw")

    # Add a text label on top of the aberration window
    text_label = ttk.Label(content, text="Enter Aberration Values in units nm (RMS value):")
    text_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

    # Create a list to store the entry points
    entry_points = []
    names = ['Oblique astigmatism', 'Vertical astigmatism', 'Vertical coma', 'Horizontal coma', 'Vertical trefoil',
             'Oblique trefoil', 'Primary spherical', 'Vertical secondary astigmatism', 'Oblique secondary astigmatism',
             'Vertical quadrafoil', 'Oblique quadrafoil']

    # Create and place the entry points in the content frame
    for i in range(int(val)):
        n, m = noll_to_zern(i + 5)
        if i < len(names):
            label = ttk.Label(content, text=f"n={n}, m={m}, {names[i]}:")
        else:
            label = ttk.Label(content, text=f"n={n}, m={m}:")
        label.grid(row=i + 1, column=0, padx=10, pady=5)

        entry = ttk.Entry(content)
        entry.insert(0, "0.0")  # Set the default value to zero
        entry.grid(row=i + 1, column=1, padx=10, pady=5)
        entry_points.append(entry)

    # Create and place the save button
    save_button = ttk.Button(content, text="Save as .npy, make sure to load and use the aberration file afterwards",
                             command=save_aberration_values)
    save_button.grid(row=int(val) + 2, column=0, columnspan=2, padx=10, pady=10)
    # Create and place the "Use Aberration Values" button
    use_button = ttk.Button(content, text="Use Aberration Values", command=use_aberration_values)
    use_button.grid(row=int(val) + 4, column=0, columnspan=2, padx=10, pady=10)

    # Create and place the "Load Aberration Values" button
    load_button = ttk.Button(content, text="Load previously saved Aberration Values",
                             command=lambda: load_aberration_values(entry_points))
    load_button.grid(row=int(val) + 3, column=0, columnspan=2, padx=10, pady=10)

    # Update the Canvas scroll region when the content changes
    content.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))
    root.wait_window(aberration_window)
@handle_exceptions
def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# @handle_exceptions
# def fit_aberrations():
#     pass


# @handle_exceptions
# def load_aberrations_file():
#     # Code to load aberrations file
#     pass



@handle_exceptions
def compute_imaging_depth():
    global params_dict
    #z_pos = float(entry_z_pos.get())
    load_variables_as_params(entries)

    params, zvals= api_gui(params_dict)

    entry_z_pos.delete(0, tk.END)
    entry_z_pos.insert(0, zvals[0]-zvals[1])


###################################################################################################################################################33

root.title("Parameter Configuration")
root.geometry("900x900")
#
#
desired_image_size = (300, 200)  # Specify the width and height in pixels

# Load the image file
image = Image.open("./fig_imagingdepth.png")
resized_image = image.resize(desired_image_size, Image.LANCZOS)

# Create a PhotoImage object from the image
photo = ImageTk.PhotoImage(resized_image)


default_values = [
    30,  # Max Iterations for Fit
    1e-6,  # Tolerance
    "cuda",  # Device
    1.49,  # NA
    1.51,  # Refractive index medium
    1.51,  # Refractive index coverglass
    1.51,  # Refractive index immersion oil
    120000,  # Free working distance obj. (nm)
    -300,  # min z-Range of bead stack (nm)
    300,  # max z-Range of bead stack (nm)
    -300,  # min z-spread of estimator (nm)
    300,  # max z-spread of estimator (nm)
    710,  # emission wavelength (nm)
    52,  # Pupil size in pixels
    65,  # Pixel Size in image plane (nm)
    16,  # Roisize in pixels
    12,  # number of aberrations
    20000,  # batch size
    1e-3,  # damping factor
    256,
    256

]

# Z-Stack Checkbox
check_zstack = tk.IntVar()
check_zstack.set(0)
zstack_checkbox = ttk.Checkbutton(root, text="Z-Stack (check this if the stage moves w.r.t. the objective)", variable=check_zstack)
zstack_checkbox.grid(column=0, row=0, columnspan=3)
zstack_checkbox.config(command=toggle_zstack)
@handle_exceptions
def validate_numeric_input(input_value):
    pattern = r'^[-+]?[0-9]*\.?[0-9]+$'  # Regular expression to match a number
    if re.match(pattern, input_value):
        return True
    else:
        return False


# Label and Entry Field Configuration
labels = [
    "Max Iterations for Fit [int]",
    "Tolerance [float]",
    "Device ['cuda' or 'cpu']",
    "NA [float]",
    "Refractive index medium [float]",
    "Refractive index coverglass [float]",
    "Refractive index immersion oil [float]",
    "Free working distance obj. (nm) [float]",
    "min z-Range of bead stack (nm) [float]",
    "max z-Range of bead stack (nm) [float]",
    "min z-spread of estimator (nm) [float]",
    "max z-spread of estimator (nm) [float]",
    "emission wavelength (nm) [float]",
    "Pupil size in pixels [even int]",
    "Pixel Size in image plane (nm) [float]",
    "Roisize in pixels [even int]",
    "Number of Zernike aberrations to fit/use \n (start from Noll index=5) [int]",
    "Batch size for fitting [int] (to prevent memory issues)",
    "Damping factor LM algorithm [float]",
    "Width image (only relevant for SMLM) [int]",
    "Height image (only relevant for SMLM) [int]"
]

# Create a label to display the image
image_label = ttk.Label(root, image=photo)
image_label.grid(column=2, row=8, columnspan=2,rowspan=8, padx=10, pady=10, sticky="nw")

# Image Caption Label
caption_label = ttk.Label(root, text="See the figure above for the definition for\n z_stage and imaging depth z_d.")
caption_label.grid(column=2, row=8+8, columnspan=2, padx=10, sticky="nw")

# Z-Pos Entry Field
label_imaging_depth = ttk.Label(root, text="Imaging depth (nm)")
label_imaging_depth.grid(column=0, row=len(labels) + 3, sticky="e")
entry_imaging_depth = ttk.Entry(root)
entry_imaging_depth.grid(column=1, row=len(labels) + 3, sticky="w")

# Compute Imaging Depth Button
btn_compute_depth = ttk.Button(root, text="press to compute z_stage from imaging depth \n  after filling in other values. \n Set the depth such that z-stage \n "
                                          "corresponds with your experiment", command=compute_imaging_depth)
btn_compute_depth.grid(column=2, row=len(labels) + 3, sticky="e")

# Imaging Depth Entry Field
label_z_pos = ttk.Label(root, text="z-stage (nm)")
label_z_pos.grid(column=0, row=len(labels) + 4, sticky="e")
entry_z_pos = ttk.Entry(root)
entry_z_pos.grid(column=1, row=len(labels) + 4, columnspan=2, sticky="w")

entries = {}
for i, label in enumerate(labels):
    ttk.Label(root, text=label).grid(column=0, row=i + 3, padx=10, pady=5, sticky="e")
    entry = ttk.Entry(root)
    entry.insert(0, default_values[i])  # Set default value
    entry.grid(column=1, row=i + 3, padx=10, pady=5, sticky="w")
    entries[label] = entry

# Get the entry fields for zrange and zspread
#entry_imaging_depth = entries["Imaging depth (nm)"]
entry_zrange_min = entries["min z-Range of bead stack (nm) [float]"]
entry_zrange_max = entries["max z-Range of bead stack (nm) [float]"]

# Disable zrange and zspread entry fields initially
entry_zrange_min.config(state=tk.DISABLED)
entry_zrange_max.config(state=tk.DISABLED)
# Pick your own zernike polynomial aberrations button
load_button = ttk.Button(root, text="Pick your own Zernike polynomial aberrations", command=lambda: choose_own_aberrations(entries["Number of Zernike aberrations to fit/use \n (start from Noll index=5) [int]"].get()))
load_button.grid(column=0, row=len(labels) + 5, columnspan=2, padx=10, pady=5, sticky="we")

# Load aberration button
# load_aberration_button = ttk.Button(root, text="Load Aberration File", command=load_aberration_file)
# load_aberration_button.grid(column=0, row=len(labels) + 6, columnspan=2, padx=10, pady=5, sticky="we")

# Load parameters button
load_button = ttk.Button(root, text="Load Parameters", command=load_params_from_file)
load_button.grid(column=0, row=len(labels) + 6, columnspan=2, padx=10, pady=5, sticky="we")

# Save parameters button
save_button = ttk.Button(root, text="Save Parameters", command=save_params_to_file)
save_button.grid(column=0, row=len(labels) + 7, columnspan=2, padx=10, pady=5, sticky="we")

# Submit button
# Define the styles
style = ttk.Style()


# Configure the style for the submit button
style.configure("SubmitButton.TButton", font=("Helvetica", 12, "bold"), relief="raised", background="#0074D9", foreground="white")
style.map("SubmitButton.TButton", background=[("active", "#0056B3")], foreground=[("active", "white")])

# Submit button
submit_button = ttk.Button(root, text="Finished? Make sure to fill everything in and \ncontinue here for more options! \n"
                                      "1. Simulation\n"
                                      "2. Estimation"
                                      ,
                           command=submit, style="SubmitButton.TButton")

submit_button.grid(column=0, row=len(labels) + 8, columnspan=2,rowspan=4, padx=10, pady=10, sticky="we")


root.mainloop()

