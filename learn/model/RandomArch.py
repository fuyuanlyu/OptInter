import numpy as np
import os

# arch_parameters = np.random.rand(325, 3)
# save_path = 'logs/search-Criteo-random-10'
# os.makedirs(save_path, exist_ok=True)
# np.save(os.path.join(save_path, 'arch_weight.npy'), arch_parameters)


# arch_parameters = np.random.rand(276, 3)
# save_path = 'logs/search-Avazu-random-10'
# os.makedirs(save_path, exist_ok=True)
# np.save(os.path.join(save_path, 'arch_weight.npy'), arch_parameters)


arch_parameters = np.random.rand(120, 3)
save_path = 'logs/search-iPinYou-random-10'
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, 'arch_weight.npy'), arch_parameters)
