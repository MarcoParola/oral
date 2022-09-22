import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from matplotlib import pyplot as plt


class Triplet_Oral_Dataset(Dataset):
    '''
    The sets of data are prepared differently depending on whether it is train, validation or test

    TRAIN: For each sample (anchor) randomly chooses a positive and negative samples
    TEST and VALIDATION: Creates fixed triplets for testing
    '''

    def __init__(self, json_dataset_file, images_path, train=False, transform=None):

        self.json_dataset_file = json_dataset_file
        self.images_path = images_path
        self.train = train
        self.transform = transform
    
        # Opening JSON file 
        with open(json_dataset_file) as json_file:
            data = json.load(json_file)

        self.labels = [item['label'] for item in data['data']]
        self.bboxes = [item['bbox'] for item in data['data']]

        # train
        if self.train:
            self.train_data = [item['image_name'] for item in data['data']]
            self.labels_set = set(np.array(self.labels))
            # generate random triplets for training
            self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                     for label in self.labels_set}

        # test or validation
        else:
            self.test_data = [item['image_name'] for item in data['data']]
            # generate fixed triplets for testing
            self.labels_set = set(np.array(self.labels))
            self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState()

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i]]),
                         random_state.choice(self.label_to_indices[
                                np.random.choice(list(self.labels_set - set([self.labels[i]]))
                            )])]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    

    def __getitem__(self, index):

        # return images in case of training set
        if self.train:
            img1 = read_image(os.path.join(self.images_path, self.train_data[index]))
            label1 = self.labels[index]
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = read_image(os.path.join(self.images_path, self.train_data[positive_index]))
            img3 = read_image(os.path.join(self.images_path, self.train_data[negative_index]))
            # crop images using the bounding boxes
            img1 = self.crop_image(img1, list(map(int, self.bboxes[index] )) )
            img2 = self.crop_image(img2, list(map(int, self.bboxes[positive_index] )))
            img3 = self.crop_image(img3, list(map(int, self.bboxes[negative_index] )))

        # return images in case of test set or validation set
        else:
            img1 = read_image(os.path.join(self.images_path, self.test_data[self.test_triplets[index][0]]))
            img2 = read_image(os.path.join(self.images_path, self.test_data[self.test_triplets[index][1]]))
            img3 = read_image(os.path.join(self.images_path, self.test_data[self.test_triplets[index][2]]))
            # crop images using the bounding boxes
            img1 = self.crop_image(img1, list(map(int, self.bboxes[self.test_triplets[index][0]] )) )
            img2 = self.crop_image(img2, list(map(int, self.bboxes[self.test_triplets[index][1]] )))
            img3 = self.crop_image(img3, list(map(int, self.bboxes[self.test_triplets[index][2]] )))
        
        # apply transformation to each images
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        
        return (img1, img2, img3), []


    # utility function to get the portion of image containing the lesion
    def crop_image(self, img, bbox):
        return img[:, bbox[1] : bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2] ]
    

    def __len__(self):
        return len(self.labels)
    


# testing main
if __name__ == '__main__':
    train = Triplet_Oral_Dataset('/home/marco/Documents/oral/data/oral_dataset/train.json', '/home/marco/Documents/oral/data/oral_dataset/images/', train=False)
    print(train.__len__())
    
    for i in range(10):
        imgs, lbl = train.__getitem__(i)
        plt.imshow(  imgs[0].permute(1, 2, 0)  )
        plt.show()  
        plt.imshow(  imgs[1].permute(1, 2, 0)  )
        plt.show()  
        plt.imshow(  imgs[2].permute(1, 2, 0)  )
        plt.show()  
    