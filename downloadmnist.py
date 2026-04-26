import requests 
import os

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60K training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10K test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60K training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10K test labels.
}

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)


base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

for filename in data_sources.values(): 
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath): 
        print("Downloading file", filename)
        response = requests.get(base_url + filename, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f: 
            for chunk in response.iter_content(chunk_size=128): 
                f.write(chunk)




