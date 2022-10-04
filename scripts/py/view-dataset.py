import hydra
import os
from src.datasets import get_datasets
from matplotlib import pyplot as plt
from hydra.utils import get_original_cwd
from tqdm import trange

def save_dataset(name, ds, elements):
    folder = os.path.join(get_original_cwd(), "tmp/dataset", name)
    os.makedirs(folder, exist_ok=True)
    
    for i in trange(0, elements):
        (a, p, n), _ = ds[i]
        fig, axes = plt.subplots(1, 3)
        plt.tight_layout()
        
        axes[0].imshow(a.permute(1, 2, 0))
        axes[0].set_title("anchor")

        axes[1].imshow(p.permute(1, 2, 0))
        axes[1].set_title("positive")

        axes[2].imshow(n.permute(1, 2, 0))
        axes[2].set_title("negative")

        plt.savefig(os.path.join(folder, f"{i}.png"))
        plt.close()

@hydra.main(config_path="../../config/", config_name="config")
def main(cfg):
    train_set, validation_set, test_set = get_datasets(cfg)
    save_dataset("train", train_set, 10)
    save_dataset("val", validation_set, 10)
    save_dataset("test", test_set, 10)
    
    
if __name__ == "__main__":
    main()