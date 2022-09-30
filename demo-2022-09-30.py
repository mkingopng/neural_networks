"""
DESCRIPTION

Copyright (C) Weicong Kong, 30/09/2022
"""

import glob

exts = ['md', 'csv', 'pdf']
for ext in exts:
	files = glob.glob(f'./**/*.{ext}', recursive=True)
	print(files)
