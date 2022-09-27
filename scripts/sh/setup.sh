mkdir models
python3 -m venv env
. env/bin/activate
python -m pip install -r requirements.txt

# TODO download data

#python train.py datasets.filenames.dataset=oral1-10.json project_path=$(pwd)
#python test.py datasets.filenames.dataset=oral1-10.json project_path=$(pwd)
