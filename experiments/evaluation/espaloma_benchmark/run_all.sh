# script that evaluates a grappa model on the espaloma dataset and creates a table of the results, stored as table.png

RUN_ID=mjolg4yf

set -e

python make_data_dict.py $RUN_ID
python make_table_dict.py
python make_tex_table.py

pdflatex table.tex table.tex

rm table.aux table.log

pdftoppm -png -r 500 table.pdf table

rm table.pdf