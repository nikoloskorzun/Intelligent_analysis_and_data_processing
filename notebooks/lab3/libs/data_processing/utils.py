from ipywidgets import interact, widgets
import os
from IPython.display import display
import numpy as np
import pandas as pd

from .dataset import Dataset

def load_datasets( path="../datasets"):
    names = {}
    for datasets_path in os.listdir(os.path.abspath(os.path.join(os.getcwd(), path))):
        names[datasets_path] = os.path.abspath(os.path.join(os.getcwd(), path, datasets_path))

    datasets_selector = widgets.SelectMultiple(
        options=names, 
        description='Датасеты:',
        layout={'width': '500px'}
    )

    
    # Кнопка для подтверждения выбора
    button = widgets.Button(description="ОК")
    
    # Выходная область для отображения результата
    output = widgets.Output()
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_array = (datasets_selector.value)
            return selected_array
    
    button.on_click(on_button_clicked)
    
    # Отображаем виджеты
    display(datasets_selector, button, output)
    
    # Чтобы можно было получить значение через return, сохраняем его

    return datasets_selector



def print_results(results):
    for el in results:
        print(f"Датасет [{el[1]}] метод [{el[2]}] на выборке [{el[3]}] дал результаты {el[0]}")

def print_scores(scores):
    s = ""
    for el in scores:
        s+=f"{el} = {scores[el]:5.4f} "
    return s

def interactive_grouper(results):
    """Интерактивная группировка данных из списка results."""
    @interact(group_by={
        'Датасет': 1,
        'Метод': 2,
        'Выборка': 3
    })
    def _group_and_show(group_by):
        groups = {}
        for el in results:
            key = el[group_by]
            groups.setdefault(key, []).append(el)
        
        for group_name, items in groups.items():
            print(f"\nГруппа [{group_name}]:")
            for item in items:
                temp = [f"Датасет [{item[1]}] ",
                    f"Метод [{item[2]}] ",
                    f"Выборка [{item[3]}] ",
                    f"→ Результат: {print_scores(item[0])}"]
                temp_len = 0
                for i in range(len(temp)):
                    if i==group_by-1:
                        continue
                    if i == 3:
                        print(" "*(66-temp_len), end="")
                    temp_len+=len(temp[i])
                    print(temp[i], end="")
                print()
        print("\n" + "="*60 + "\n")