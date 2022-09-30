# oral

clone repo
```sh
git clone ...
cd ...
mkdir data/oral_dataset/images
```

download images and coco dataset


```sh
wget ... # remove directory
unzip data/oral_dataset/oral1.zip -d data/oral_dataset/images/
rm -r data/oral_dataset/images/*/    
```

create and activate virtual environment, then install dependencies
```sh
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt 
```

split dataset
```
python -m scripts.py.split_dataset
```

train and test
```
python train.py datasets.filenames.dataset=oral1-11.json project_path=$(pwd)
python test.py datasets.filenames.dataset=oral1-11.json project_path=$(pwd)
```