import re

name = "./images/lena.bmp"

regex = re.compile(r'(?!images\b)\b(?!bmp\b)\b\w+')

file_name = regex.search(name)
print(file_name.group())

name = re.compile(r'(?!images\b)\b(?!bmp\b)\b\w+').search(name).group()