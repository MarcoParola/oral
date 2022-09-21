mkdir data/oral_dataset/images
unzip data/oral_dataset/oral1.zip -d data/oral_dataset/images/
rm -r data/oral_dataset/images/*/    # remove directory

# rm data/oral_dataset/oral1.zip

python -m scripts.py.split_dataset datasets.filenames.dataset=oral1-10.json datasets.path=$(pwd)/data/oral_dataset/
