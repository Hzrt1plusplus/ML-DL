import matplotlib.pyplot as plt 
import numpy as np 
import os
import gzip 


files = {
    "train_images" : "train-images-idx3-ubyte.gz", 
    "train_labels" : "train-labels-idx1-ubyte.gz", 
    "test_images" : "t10k-images-idx3-ubyte.gz",
    "test_labels" : "t10k-labels-idx1-ubyte.gz"
}

data = {}

datadir = "./data"

for key in ("test_images", "train_images"):
    with gzip.open(os.path.join(datadir, files[key]), "rb") as file: 
        data[key] = np.frombuffer(file.read(), offset = 16, dtype = np.uint8).reshape(-1, 28*28)
        
for key in ("test_labels", "train_labels"): 
    with gzip.open(os.path.join(datadir, files[key]), "rb") as file: 
        data[key] = np.frombuffer(file.read(), offset = 8, dtype = np.uint8)


def print_image(img): 

    for i in range(28*28): 
        if i % 28 == 27: 
            print()
        if img[i] > 100: 
            print("█",end="")
        else: 
            print(" ",end="")

index = 0 
def show(): 

    global index

    i1 = data["train_images"][index]
    index += 1 
    print_image(i1)
    i1 = i1.reshape((28,28))

    plt.imshow(i1, cmap = "gray")
    plt.show()


while True: 

    a = input()
    if a == "exit": 
        break 

    show()

