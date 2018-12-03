import numpy as np

def generate_data():
    """Generate binary labelled data of the form [0, num, num, 0] and [num, num, 0, 0]"""
    data = []
    labels = []
    for i in range (0, 50 ):
        x = np.random.random_sample()
        data.append([x, 1-x, 0, 0])
        labels.append(0)

    for j in range(50 , 100  ):
        x = np.random.random_sample()
        data.append([0, x, 1-x, 0])
        labels.append(1)

    np.savetxt("data/data.csv", data, delimiter = ",")
    np.savetxt("data/labels.csv", labels, delimiter = ",")

def read_data():
    print(np.loadtxt(open("data/data.csv", "rb"), delimiter = ","))
    print(np.loadtxt(open("data/labels.csv", "rb"), delimiter = ","))


if __name__ == "__main__":
    generate_data()
    read_data()
