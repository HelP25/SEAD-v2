import pandas as pd
import numpy as np
import matplotlib as mpl

# df = pd.DataFrame([[38.0, 2.0, 18.0, 22.0, 21, np.nan],[19, 439, 6, 452, 226,232]],
#                   index=pd.Index(['Tumour (Positive)', 'Non-Tumour (Negative)'], name='Actual Label:'),
#                   columns=pd.MultiIndex.from_product([['Decision Tree', 'Regression', 'Random'],['Tumour', 'Non-Tumour']], names=['Model:', 'Predicted:']))
# df.style
# s = df.style.format('{:.0f}').hide([('Random', 'Tumour'), ('Random', 'Non-Tumour')], axis="columns")
#
# cell_hover = {  # for row hover use <tr> instead of <td>
#     'selector': 'td:hover',
#     'props': [('background-color', '#ffffb3')]
# }
# index_names = {
#     'selector': '.index_name',
#     'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
# }
# headers = {
#     'selector': 'th:not(.index_name)',
#     'props': 'background-color: #000066; color: white;'
# }
# s.set_table_styles([cell_hover, index_names, headers])
#
# out = s.set_table_attributes('class="my-table-cls"').to_html()
# print(out[out.find('<table'):][:109])
