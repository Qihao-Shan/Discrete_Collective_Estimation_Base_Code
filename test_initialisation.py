import numpy as np
import matplotlib.pyplot as plt
from arena_class import arena

fig, axis = plt.subplots(2, 2)
hypotheses = np.array([0.1, 0.3, 0.7, 0.9])
a = arena(hypotheses, dm_strat='DC', axis=axis)
a.plot_arena(0)
plt.show()