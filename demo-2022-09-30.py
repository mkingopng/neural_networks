"""
DESCRIPTION

Copyright (C) Weicong Kong, 30/09/2022
"""

import glob
import pandas as pd

exts = ['md', 'csv', 'pdf']
for ext in exts:
	files = glob.glob(f'./**/*.{ext}', recursive=True)
	print(files)


# WK: read_excel solves a lot of excel reading problem
excel_df = pd.read_excel()


# WK: num of week in a year

import datetime

week_num = 38
year = 2022
weekday = 0  # Sunday
week_start_at = datetime.datetime.strptime(f'{year}-{week_num}-{weekday}', "%Y-%W-%w")
print(week_start_at)