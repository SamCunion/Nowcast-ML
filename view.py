#dataset entry viewer
import matplotlib.pyplot as plot
import pandas as pd
import os
import sys
from tornet.data.loader import read_file
from tornet.display.display import plot_radar, plot_grid
from dotenv import load_dotenv
load_dotenv()

#options
DATASET_PATH = os.getenv("DATASET_PATH")
PLOTS = ["DBZ", "VEL", "RHOHV"]

def get_random_entry(catalogue):
    random_item = catalogue.sample(n=1)
    filename = random_item["filename"].values[0]
    if os.path.exists(DATASET_PATH + "/" + filename):
        return random_item
    else:
        print("Unable to find file: " + filename)
        return get_random_entry(catalogue)


def change_displayed_info(fg, entry):
    filename = entry["filename"].values[0]
    file = read_file(DATASET_PATH + "/" + filename)
    plot_radar(file, fig=fg, channels=PLOTS, include_cbar=False, time_idx=-1, n_rows=2, n_cols=3)
    plot_grid(file, fig=fg)
    ef = str(entry["ef_number"].values[0])
    type = entry["category"].values[0]
    datetime = entry["start_time"].values[0]
    traintest = entry["type"].values[0]

    fg.text(.5, .05, "Batch: " + traintest + ", Type: " + type + ", EF: " + ef + ", Datetime: " + datetime, ha="center")

def on_keypress(e):
    if (e.key == "escape"):
        plot.close()
        sys.exit()
    figure.clear()
    rand_entry = get_random_entry(catalogue)
    change_displayed_info(figure, rand_entry)
    figure.canvas.draw()
        


#entrypoint
if __name__ == "__main__":
    #read the catalogue
    catalogue = pd.read_csv(DATASET_PATH + "/catalog.csv")
    #init matplotlib
    figure = plot.figure(figsize=(12, 4))
    #link key event
    figure.canvas.mpl_connect("key_press_event", on_keypress)

    #get starter display
    change_displayed_info(figure, get_random_entry(catalogue))

    plot.show()