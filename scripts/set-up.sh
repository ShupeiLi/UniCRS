#!/bin/bash
data_path="$(pwd)/data/dbpedia"

git pull origin main
cp "$HOME/autodl-fs/dbpedia-universal.zip" "${data_path}"
unzip "${data_path}/dbpedia-universal.zip" -d "${data_path}"
rm "${data_path}/dbpedia-universal.zip"
python data/dbpedia/extract_kg.py
