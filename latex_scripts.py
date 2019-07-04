latex = """
\\begin{tabularx}{\\linewidth}{ | c | @{}X@{} | } 
    \\hline
    \\multicolumn{2}{|c|}{\textbf{Results}} \\\\
    \\hline
    \\textbf{Algorithm} & \\multicolumn{1}{|c|}{\textbf{Measure \\& Configuration}}\\\\
    \\hline
    Felzenszwalb &
    {\\begin{tabularx}{\\linewidth}{c|X}
        f-score  & $scale = ?$, $sigma = ?$, $min\\_size = ?$ \\\\
        precision & $scale = ?$, $sigma = ?$, $min\\_size = ?$\\\\
        recall & $scale = ?$, $sigma = ?$, $min\\_size = ?$ \\\\
        \texorpdfstring{f\\textsubscript{0.05}}-score & $scale = ?$, $sigma = ?$, $min\\_size = ?$ \\\\
    \end{tabularx}}
    \tabularnewline
    \hline
    Slic &
    {\\begin{tabularx}{\\linewidth}{c|X}
        f-score  & $n\\_segments = ?$, $compactness = ?$, $sigma = ?$, $min\\_size\\_factor = ?$\\\\
        precision & $n\\_segments = ?$, $compactness = ?$, $sigma = ?$, $min\\_size\\_factor = ?$\\\\
        recall & $n\\_segments = ?$, $compactness = ?$, $sigma = ?$, $min\\_size\\_factor = ?$\\\\
        \texorpdfstring{f\\textsubscript{0.05}}-score & $n\\_segments = ?$, $compactness = ?$, $sigma = ?$, $min\\_size\\_factor = ?$\\\\
    \\end{tabularx}}
    \\tabularnewline
    \\hline
    Watershed & 
    {\\begin{tabularx}{\\linewidth}{c|X}
        f-score & $markers = ?$, $compactness = ?$, $gradient = ?$ \\\\
        precision & $markers = ?$, $compactness = ?$, $gradient = ?$ \\\\
        recall & $markers = ?$, $compactness = ?$, $gradient = ?$ \\\\
        \\texorpdfstring{f\\textsubscript{0.05}}-score & $markers = ?$, $compactness = ?$, $gradient = ?$ \\\\
    \\end{tabularx}}
    \\tabularnewline
    \\hline
    Fast Scanning & 
    {\\begin{tabularx}{\\linewidth}{c|X}
        f-score & $max\\_diff = ?$, $min\\_size\\_factor = ?$ \\\\\\
        precision & $max\\_diff = ?$, $min\\_size\\_factor = ?$ \\\\
        recall & $max\\_diff = ?$, $min\\_size\\_factor = ?$ \\\\
        \\texorpdfstring{f\textsubscript{0.05}}-score & $max\\_diff = ?$, $min\\_size\\_factor = ?$ \\\\
    \\end{tabularx}}
    \\tabularnewline
    \\hline
    Our Algorithm: & \\
    - simple & 
    {\\begin{tabularx}{\\linewidth}{c|X}
        \\hline
        f-score & $max\\_diff = ?$, $min\\_size\\_factor = ?$ \\\\
        precision & $max\\_diff = ?$, $min\\_size\_factor = ?$ \\\\
        recall & $max\\_diff = ?$, $min\\_size\\_factor = ?$ \\\\
        \\texorpdfstring{f\textsubscript{0.05}}-score & $max\_diff = ?$, $min\_size\_factor = ?$ \\\\
        \\hline
    \end{tabularx}}
    \tabularnewline

    - variance &
    {\\begin{tabularx}{\\linewidth}{c|X}
        f-score & $max\\_diff =?$, $min\\_size\\_factor= ?$, $min\\_var = ?$\\\\
        precision & $max\\_diff =?$, $min\\_size\\_factor= ?$, $min\\_var = ?$\\\\
        recall & $max\\_diff =?$, $min\\_size\\_factor= ?$, $min\\_var = ?$\\\\
        \\texorpdfstring{f\textsubscript{0.05}}-score & $max\\_diff =?$, $min\\_size\\_factor= ?$, $min\\_var = ?$\\\\
        \\hline
    \\end{tabularx}}
    \\tabularnewline
    
    - exponent & 
    {\\begin{tabularx}{\\linewidth}{c|X}
        f-score & $max\\_diff =?$, $min\\_size\\_factor= ?$, $exponent = ?$\\\\
        precision & $max\\_diff =?$, $min\\_size\\_factor= ?$, $exponent = ?$\\\\
        recall & $max\\_diff =?$, $min\\_size\\_factor= ?$, $exponent = ?$\\\\
        \texorpdfstring{f\\textsubscript{0.05}}-score & $max\\_diff =?$, $min\\_size\\_factor= ?$, $exponent = ?$\\\\
    \\end{tabularx}}
    \\tabularnewline
    \\hline
\\end{tabularx}
"""



def best_configurations(latex_array):
    ### Creating a latex table from the results:
    import re
    pattern = r'\?'
    find = re.findall(pattern, latex)
    print(len(find))
    print(len(latex_array))
    assert len(find) == len(latex_array)
    iterator = iter(str(i) for i in latex_array)
    table = re.sub(pattern, lambda L: next(iterator), latex)
    return table


def print_values(algorithm, configurations, sort_func):
    string = "{} & {}\% & {}\% & {}\% & {}\% & {}\% \\\\ ".format(
        algorithm.title(),
        round(configurations[0]['npri_score']*100, 1), 
        round(configurations[1]['f_score']*100, 1), 
        round(configurations[2]['precision']*100, 1),
        round(configurations[3]['recall']*100, 1),
        round(sort_func(configurations[4])*100, 1)
    )
    return string


def print_best_overall(scores_overall, sort_func):
    print(f"Overall results:")
    print("Best rand-index:")
    print(sorted(scores_overall, key=lambda x: x['npri_score'], reverse=True)[0])
    print("Best f_score:")
    print(sorted(scores_overall, key=lambda x: x['f_score'], reverse=True)[0])
    print("Best precision:")
    print(sorted(scores_overall, key=lambda x: x['precision'], reverse=True)[0])
    print("Best recall:")
    print(sorted(scores_overall, key=lambda x: x['recall'], reverse=True)[0])
    print("Best custom:")
    print(sorted(scores_overall, key=sort_func, reverse=True)[0])