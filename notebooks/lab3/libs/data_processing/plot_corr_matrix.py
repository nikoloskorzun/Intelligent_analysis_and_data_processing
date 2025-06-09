import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import SelectMultiple, Dropdown, interactive_output, VBox, Output, HBox
import ipywidgets as widgets
from numba import jit
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap



@jit(nopython=True)
def rank_columns(data):
    ranked = np.empty_like(data)
    for i in range(data.shape[1]):
        column = data[:, i]
        sorted_indices = np.argsort(column)
        ranks = np.zeros_like(sorted_indices, dtype=np.float64)
        current_rank = 0
        while current_rank < len(sorted_indices):
            current_value = column[sorted_indices[current_rank]]
            mask = column == current_value
            indices = np.where(mask)[0]
            avg_rank = (current_rank + len(indices) + 1 + current_rank) / 2.0
            ranks[indices] = avg_rank
            current_rank += len(indices)
        ranked[:, i] = ranks
    return ranked

@jit(nopython=True)
def pearson_correlation_matrix(data):
    n = data.shape[1]
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            x = data[:, i]
            y = data[:, j]
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            cov = np.sum((x - mean_x) * (y - mean_y))
            std_x = np.sqrt(np.sum((x - mean_x)**2))
            std_y = np.sqrt(np.sum((y - mean_y)**2))
            if std_x == 0 or std_y == 0:
                corr = 0.0
            else:
                corr = cov / (std_x * std_y)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    return corr_matrix




def plot_interactive_correlation_heatmap(dataset):
    
    numeric_cols = dataset.select_dtypes(include=np.number).columns.tolist()

    colors = [(1, 1, 1), (0, 0, 0), (1, 1, 1)]  # Белый -> Черный -> Белый
    # Создаем цветовую карту
    corr_best_cmap = LinearSegmentedColormap.from_list('corr_best_cmap', colors, N=256)
    
    if not numeric_cols:
        print("Нет числовых столбцов для построения корреляции.")
        return

    # Виджеты
    columns_widget = SelectMultiple(
        options=numeric_cols,
        value=[numeric_cols[0]],
        description='Столбцы'
    )
    
    method_widget = Dropdown(
        options=['pearson', 'spearman', 'kendall'],
        value='pearson',
        description='Метод'
    )

    cmap_widget = Dropdown(
        options=['gray_r', 'viridis', 'coolwarm', 'plasma', 'inferno', 'magma', 'cividis', corr_best_cmap],
        value = corr_best_cmap,
        description='Палитра'
    )
    
    annot_widget = widgets.Checkbox(
        value=True,
        description='Показывать значения'
    )
    
    font_widget = widgets.Dropdown(
        options=['DejaVu Sans', 'Arial', 'Times New Roman', 'Courier New', 'Helvetica'],
        value='DejaVu Sans',
        description='Шрифт'
    )
    
    font_size_widget = widgets.IntSlider(
        value=10,
        min=4,
        max=36,
        description='Размер значений'
    )
    


    
    save_widget = widgets.Text(
        value='correlation_heatmap.png',
        description='Имя файла:'
    )
    
    size_widget = widgets.IntSlider(
        value=10,
        min=5,
        max=50,
        step=1,
        description='Размер (дюймы):'
    )
    
    dpi_widget = widgets.IntSlider(
        value=300,
        min=100,
        max=1200,
        step=100,
        description='DPI:'
    )
    
    format_widget = Dropdown(
        options=['png', 'pdf', 'svg', 'tiff', 'jpeg'],
        value='png',
        description='Формат:'
    )

    linewidth_widget = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=2,
        step=0.1,
        description='Толщина линий'
    )

    
    save_button = widgets.Button(description="Сохранить график")
    out = Output()
    
    last_params = {
        'corr': None,
        'method': 'pearson',
        'cmap': 'gray_r',
        'annot': True,
        'font': 'DejaVu Sans',
        'font_size': 10,
        'linewidth': 0.5
    }

    def update_plot(columns, method, cmap, annot, font, font_size, linewidth, figsize):
        last_params.update({
            'method': method,
            'cmap': cmap,
            'annot': annot,
            'font': font,
            'font_size': font_size,
            'linewidth': linewidth
        })
        
        out.clear_output(wait=True)
        if not columns:
            with out:
                print("Выберите хотя бы один столбец.")
            return
        
        data = dataset[list(columns)].dropna()
        if data.empty:
            with out:
                print("Нет данных после удаления пропущенных значений.")
            return
        
        plt.rcParams['font.family'] = font
        
        if method in ['pearson', 'spearman']:
            data_np = data.to_numpy().astype(np.float64)
            if method == 'spearman':
                data_np = rank_columns(data_np)
            corr_matrix = pearson_correlation_matrix(data_np)
            corr = pd.DataFrame(corr_matrix, index=data.columns, columns=data.columns)
        else:
            corr = data.corr(method)
        
        last_params['corr'] = corr
        
        with out:
            plt.figure(figsize=(figsize, figsize*0.8))
            sns.heatmap(
                corr,
                annot=annot,
                cmap=cmap,
                fmt=".2f",
                vmin=-1,
                vmax=1,
                linewidths=linewidth,
                annot_kws={"size": font_size}
            )
            # Настройка размера подписей осей
            plt.tick_params(axis='both', which='major', labelsize=font_size)
            plt.title(f'Тепловая карта корреляции ({method})')
            plt.show()

    def save_plot(_):
        if last_params['corr'] is None:
            print("Сначала постройте график!")
            return
            
        plt.rcParams['font.family'] = last_params['font']
        plt.rcParams['savefig.dpi'] = dpi_widget.value
        
        fig = plt.figure(figsize=(size_widget.value, size_widget.value*0.8))
        sns.heatmap(
            last_params['corr'],
            annot=last_params['annot'],
            cmap=last_params['cmap'],
            fmt=".2f",
            vmin=-1,
            vmax=1,
            linewidths=last_params['linewidth'],
            annot_kws={"size": last_params['font_size']}
        )
        # Настройка размера подписей при сохранении
        plt.tick_params(axis='both', which='major', 
                       labelsize=last_params['font_size'])
        plt.title(f'Тепловая карта корреляции ({last_params["method"]})')
        plt.savefig(
            save_widget.value,
            dpi=dpi_widget.value,
            bbox_inches='tight',
            format=format_widget.value
        )
        plt.close(fig)
        print(f"График сохранен: {save_widget.value}")

    save_button.on_click(save_plot)

    controls = VBox([
        HBox([columns_widget, VBox([method_widget, cmap_widget, font_widget, annot_widget])]),
        HBox([font_size_widget, linewidth_widget]),
        HBox([size_widget, dpi_widget]),
        HBox([save_widget, format_widget]),
        save_button
    ])
    
    interactive_plot = interactive_output(
        update_plot,
        {
            'columns': columns_widget,
            'method': method_widget,
            'cmap': cmap_widget,
            'annot': annot_widget,
            'font': font_widget,
            'font_size': font_size_widget,
            'linewidth': linewidth_widget,
            'figsize': size_widget
        }
    )
    
    display(VBox([controls, interactive_plot, out]))
