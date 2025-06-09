import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, interactive, interactive_output
import numba
from IPython.display import display

def plot_interactive_histograms(dataset):
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        print("DataFrame не содержит числовых столбцов.")
        return

    @numba.jit(nopython=True)
    def sturges_bins(n):
        return int(np.ceil(np.log2(n) + 1)) if n else 1

    @numba.jit(nopython=True)
    def scott_bins(data):
        n = len(data)
        if n < 2: return 1
        h = 3.5 * np.std(data) / (n ** (1/3))
        return int(np.ceil((np.max(data) - np.min(data)) / h)) if h else 1

    @numba.jit(nopython=True)
    def fd_bins(data):
        n = len(data)
        if n < 2: return 1
        sorted_data = np.sort(data)
        q25 = sorted_data[int(0.25*n)]
        q75 = sorted_data[int(0.75*n)]
        iqr = q75 - q25
        h = 2 * iqr / (n ** (1/3)) if iqr else 2 * (np.max(data)-np.min(data)) / (n ** (1/3))
        return int(np.ceil((np.max(data) - np.min(data)) / h)) if h else 1

    @numba.jit(nopython=True)
    def sqrt_bins(n):
        return int(np.ceil(np.sqrt(n))) if n else 1

    # Виджеты
    columns_selector = widgets.SelectMultiple(
        options=numeric_columns, 
        description='Столбцы:',
        layout={'width': '500px'}
    )
    
    bin_method_selector = widgets.Dropdown(
        options=[('Sturges', 'sturges'), ('Scott', 'scott'), 
                 ('FD', 'fd'), ('Квадрат', 'sqrt'), ('Ручной', 'manual')],
        value='sturges',
        description='Метод бинов:'
    )
    
    manual_bins_input = widgets.IntText(
        value=10,
        description='Количество бинов:',
        disabled=True
    )

    # Логика отображения
    def update_ui(*args):
        manual_bins_input.disabled = bin_method_selector.value != 'manual'


        
    bin_method_selector.observe(update_ui, 'value')
    ui = widgets.VBox([columns_selector, bin_method_selector, manual_bins_input])
    out = widgets.Output()

    # Отрисовка графиков
    def plot(columns, bin_method, manual_bins):
        out.clear_output(wait=True)
        with out:
            if not columns: return
            
            fig, axes = plt.subplots(len(columns), 1, 
                          figsize=(10, 5*len(columns)), 
                          squeeze=False)
            
            for ax, col in zip(axes.flatten(), columns):
                data = dataset[col].dropna().values
                if len(data) == 0: continue
                
                if bin_method == 'sturges':
                    bins = sturges_bins(len(data))
                    method_name = "Sturges"
                elif bin_method == 'scott':
                    bins = scott_bins(data)
                    method_name = "Scott"
                elif bin_method == 'fd':
                    bins = fd_bins(data)
                    method_name = "FD"
                elif bin_method == 'sqrt':
                    bins = sqrt_bins(len(data))
                    method_name = "Квадратный корень"
                else:
                    bins = max(1, min(len(data), manual_bins))
                    method_name = "Ручной"
                
                ax.hist(data, bins=max(1, bins), alpha=0.7, 
                      edgecolor='black', density=True)
                title = f'Гистограмма: {col}\nМетод: {method_name}, Биннов: {bins}'
                ax.set_title(title, pad=12)
                ax.set_xlabel('Значения', labelpad=10)
                ax.set_ylabel('Плотность', labelpad=10)
                ax.grid(alpha=0.2)
                
            plt.tight_layout()
            plt.show()

    interactive_plot = interactive_output(
        plot, 
        {'columns': columns_selector, 
         'bin_method': bin_method_selector, 
         'manual_bins': manual_bins_input}
    )
    
    display(ui, out)