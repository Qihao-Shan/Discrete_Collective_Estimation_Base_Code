import numpy as np
import matplotlib.pyplot as plt
from arena_class import arena

fig, axis = plt.subplots()
hypotheses = np.array([0.1, 0.3, 0.7, 0.9])
a = arena(0.6, 'Random', hypotheses, dm_strat='DC', axis=axis, exp_length=1, diss_length=1)

Max_step = 12000

for i in range(Max_step):
    a.random_walk()
    a.plot_arena(i)