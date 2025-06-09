from ipywidgets import widgets
import pandas as pd
import matplotlib.pyplot as plt
def plot_interactive_scatterplots(dataset):
    def scatterplots(**kwargs):
        selected_items = [item for item, selected in kwargs.items() if selected]
        if len(selected_items) < 1:
            return
        pd.plotting.scatter_matrix(dataset[selected_items], figsize=(15, 15), diagonal='kde')
        plt.show()


    widgets.interact(scatterplots, **{item: widgets.Checkbox(description=item, value=False) for item in list(dataset.columns)})