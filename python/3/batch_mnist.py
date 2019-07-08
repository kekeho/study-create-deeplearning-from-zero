from mnist import get_data, init_network
import numpy as np


def main():
    x, t = get_data()
    network = init_network()
    
    batch_size = 100
    accuracy_count = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = network.forward(x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_count += np.sum(p == t[i:i+batch_size])
    
    print(f'Accuracy: {accuracy_count / len(x) * 100}%')



if __name__ == "__main__":
    main()

