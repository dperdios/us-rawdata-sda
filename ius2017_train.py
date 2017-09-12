import os
from lip.model import SDA
import datasets.utils as dutils
from itertools import product


# Load test set
test_set = dutils.load_ius2017_test_set()

# Load train set
full_train_set = dutils.load_ius2017_train_set()

# Generate a cross valid set
valid_set, train_set = dutils.generate_cross_valid_sets(full_set=full_train_set,
                                                        valid_size=2000,
                                                        seed=123456789)

# Benchmark launch settings
batch_size = 4096
learning_rate = 0.001
num_epochs = 20
dump_percent = 10
data_dim = train_set.shape[1]
base_dir = os.path.join('networks', 'ius2017')
cp_list = range(50, 99, 5)
lmm_list = [False, True]

for lmm, cp in product(lmm_list, cp_list):
    # SDA model
    model = SDA(data_dim=data_dim, compression_percent=cp, learn_mm=lmm, base_dir=base_dir)

    # Train model
    model.train(learning_rate=learning_rate,
                train_set=train_set,
                valid_set=valid_set,
                num_epochs=num_epochs,
                batch_size=batch_size,
                dump_percent=dump_percent)
