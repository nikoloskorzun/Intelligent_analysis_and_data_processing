import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from .dataset import Dataset
from .algo import get_EMD
def plot_interactive(dataset: Dataset):
    """
    Интерактивный график с выбором столбцов и цветовой маркировкой по целевой переменной
    
    Параметры:
    dataset (Dataset): Исходный датасет
    """
    target_column = dataset.target_features[0]

    # Проверка наличия целевого столбца
    if target_column not in dataset.columns:
        raise ValueError(f"Столбец {target_column} не найден в датасете")

    # Подготовка данных
    features = dataset.columns.drop(target_column).tolist()
    colors = dataset[target_column].map({1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'brown', 7: 'black'})

    imfs_choices = list(range(11))
    # Создание виджетов
    feature_selector = widgets.SelectMultiple(
        options=features,
        description='Признаки:',
        rows=10,
        disabled=False
    )
    imfs_count = widgets.SelectMultiple(
        options=imfs_choices,
        description='imfs:',
        rows=10,
        disabled=False
    )

    # Создание контейнера для графика
    output = widgets.Output()

    # Функция обновления графика
    def update_plot(selected_features, imfs_c):
        with output:
            output.clear_output(wait=True)
            
            plt.figure(figsize=(12, 12))
            x = range(1, len(dataset)+1)
            
            # Маркеры для разных признаков
            markers = ['.', 'o', 's', '^', 'v', '<', '>', 'p', '*', 'X', 'D']
            
            for i, feature in enumerate(selected_features):
                plt.subplot(2, 1, 1)
                plt.scatter(
                    x, 
                    dataset[feature], 
                    c=colors,
                    marker=markers[i % len(markers)],
                    alpha=0.7,
                    label=feature
                )

                plt.subplot(2, 1, 2)
                emds = get_EMD(dataset[feature].to_numpy())
                sum_imfs = np.zeros(len(emds[0]))
                if len(imfs_c) > 1:
                    for j in imfs_c:
                        if 0 <= j < len(emds):
                            sum_imfs += emds[j]
                    plt.scatter(
                        x, 
                        sum_imfs, 
                        c=colors,
                        marker='.',
                        alpha=0.7,
                        label=f"sum",
                        linewidth=0.1
                        )
                    continue
                for j, imfs in enumerate(emds):
                    if j not in imfs_c:
                        continue
                    if 0 <= j < len(emds):
                        plt.scatter(
                        x, 
                        imfs, 
                        c=colors,
                        marker='.',
                        alpha=0.3,
                        label=f"imfs {j}",
                        linewidth=0.1
                        )
            plt.subplot(2, 1, 1)
            plt.title(f'Интерактивный график признаков (Целевая переменная: {target_column})')
            plt.xlabel('Номер наблюдения')
            plt.ylabel('Значение признака')
            plt.grid(True)
            
            if selected_features:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.show()

    # Связывание виджетов с функцией
    widgets.interactive_output(update_plot, {'selected_features': feature_selector, 'imfs_c':imfs_count})
    
    # Отображение элементов
    display(widgets.VBox([widgets.HBox([feature_selector, imfs_count]), output]))