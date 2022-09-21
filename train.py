#from src.datasets import Triplet_Oral
import hydra

@hydra.main(config_path="./config/", config_name="config")
def train(cfg):
    print(cfg)

    train_json_filename = cfg.datasets.filenames.train
    val_json_filename = cfg.datasets.filenames.val
    test_json_filename = cfg.datasets.filenames.test
    img_folder = cfg.datasets.img_path
    data_path = cfg.datasets.path

    train_json_filename = os.path.join(data_path, train_json_filename)
    val_json_filename = os.path.join(data_path, val_json_filename)
    test_json_filename = os.path.join(data_path, test_json_filename)
    img_folder = os.path.join(data_path, img_folder)

    train = Triplet_Oral_Dataset(train_json_filename, img_folder, train=True)
    val = Triplet_Oral_Dataset(val_json_filename, img_folder)
    test = Triplet_Oral_Dataset(test_json_filename, img_folder)



if __name__ == '__main__':
    train()