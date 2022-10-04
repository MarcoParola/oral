import json
import os
import cv2
import hydra
import random
from collections import Counter

labels_dict = [ {'id':1, 'categories':[10]}, 
                {'id':2, 'categories':[16, 12, 11]}, 
                {'id':3, 'categories':[13,14,15] } ]


@hydra.main(config_path="../../config/", config_name="config")
def split_dataset(cfg):
    print('SPLIT')

    dataset = []
   
    dataset_json_filename = cfg.datasets.filenames.dataset
    train_json_filename = cfg.datasets.filenames.train
    val_json_filename = cfg.datasets.filenames.val
    test_json_filename = cfg.datasets.filenames.test
    data_path = cfg.datasets.path
    train_size_percentage = cfg.prepare_dataset.train_size
    val_size_percentage = cfg.prepare_dataset.val_size

    # Opening JSON file 
    with open(os.path.join(data_path, dataset_json_filename)) as json_file:
        data = json.load(json_file)

        for i in range(len(data['images'])):

            bboxes = [item for item in data['annotations'] if item['image_id'] == data['images'][i]['id']]
            for j in range(len(bboxes)):
                if '.' in data['images'][i]['path'].split('/')[3]:
                    dataset.append(
                        {'image_name': data['images'][i]['path'].split('/')[3],
                        'label': [item['id'] for item in labels_dict if bboxes[j]['category_id'] in item['categories']][0],
                        'bbox': bboxes[j]['bbox']})
    
    random.seed(cfg.prepare_dataset.seed)
    random.shuffle(dataset)

    # compute the size of each set
    train_size = int(train_size_percentage * len(dataset))
    val_size = int(val_size_percentage * len(dataset))

    # split dataset
    training_set = dataset[:train_size]
    validation_set = dataset[train_size:val_size]
    test_set = dataset[val_size:]

    print('Class distribution')
    print('train', Counter([item['label'] for item in training_set]))
    print('val', Counter([item['label'] for item in validation_set]))
    print('test', Counter([item['label'] for item in test_set]))

    # write the 3 sets of data splitted
    with open(os.path.join(data_path, train_json_filename), 'w') as json_file:
        json.dump({'data':training_set}, json_file)
    with open(os.path.join(data_path, val_json_filename), 'w') as json_file:
        json.dump({'data':validation_set}, json_file)
    with open(os.path.join(data_path, test_json_filename), 'w') as json_file:
        json.dump({'data':test_set}, json_file)


        


if __name__ == '__main__':
    split_dataset()