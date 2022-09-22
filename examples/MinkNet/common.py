import jittor as jt
import numpy as np
import random

def seed_all(random_seed):
    jt.misc.set_global_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)