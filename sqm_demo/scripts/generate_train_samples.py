# We use 700 simulations to make sure we get above 1PB

import numpy as np

x=np.random.uniform(0.49, 0.51, size=(700,))
np.random.shuffle(x)
xs = np.array_split(x, 200)
for i, ix in enumerate(xs):
  with open(f'input_data/spreads_{i:03d}', 'w') as file:
    for x_j in ix:
      file.write(f"{x_j}\n")

