#!/bin/bash

echo "Installing required packages..."
yes | pip install nbmerge

BASE_PATH="/home/adamsky/Laboratory/cvdl-for-home"
NBCONVERT_TEMPLATE="/home/adamsky/.jupyter/hidecode.tplx"

echo "Merging all Chapters into single Notebook..."
cd $BASE_PATH"/Chapters"
nbmerge ../Index.ipynb 01.Introduction.ipynb 02.LiteratureReview.ipynb 03.SystemDesign.ipynb 04.DataCollection.ipynb 05.Forecasting.ipynb 06.AnomalyDetection.ipynb 07.Conclusions.ipynb 08.References.ipynb 09.Acknowledgements.ipynb 10.Appendices.ipynb > ../Merged/Thesis.ipynb

echo "Converting Notebook to pdf..."
cd $BASE_PATH"/Merged"
jupyter nbconvert --to pdf Thesis.ipynb --template=$NBCONVERT_TEMPLATE --no-input --output Thesis.pdf
