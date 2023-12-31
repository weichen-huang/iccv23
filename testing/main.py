from testing.test_encoders import test as encoders_test
from testing.test_losses import test as losses_test
from testing.test_dataset import test as dataset_test
from dataset.utils import get_tabular_data
from logger import log

import os

def main():
    log("Begin testing")
    log("Current directory: " + os.getcwd())
    encoders_test()
    losses_test()
    # get_tabular_data(verbose=True)
    dataset_test()
    log("End testing")

if __name__ == '__main__':
    main()