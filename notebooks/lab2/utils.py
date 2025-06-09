from ipywidgets import interact

def print_results(results):
    for el in results:
        print(f"Датасет [{el[1]}] метод [{el[2]}] на выборке [{el[3]}] дал результаты {el[0]}")

def print_scores(scores):
    s = ""
    for el in scores:
       s+=f"{el} = {scores[el]:11.10f} "
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
