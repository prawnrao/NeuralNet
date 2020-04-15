import numpy as np
import sys
sys.path.insert(1, 'Users/pranavrao/Documents/playground/doodle/')


def select_n_samples(arr, n):
    arr = arr[np.random.choice(arr.shape[0], n, replace=False)]
    return np.array([a.flatten() for a in arr])


def main():
    airplanes = np.load('doodle/data/airplanes.npy')
    dogs = np.load('doodle/data/dogs.npy')
    guitars = np.load('doodle/data/guitars.npy')

    airplanes = select_n_samples(airplanes, 200)
    dogs = select_n_samples(dogs, 200)
    guitars = select_n_samples(guitars, 200)

    np.save('doodle/data/airplanes200.npy', airplanes)
    np.save('doodle/data/dogs200.npy', dogs)
    np.save('doodle/data/guitars200.npy', guitars)


if __name__ == '__main__':
    main()
