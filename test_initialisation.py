import numpy as np
import matplotlib.pyplot as plt
from arena_class import arena

fig, axis = plt.subplots(2,2)
hypotheses = np.array([0.1, 0.3, 0.7, 0.9])
a = arena(0.4, 'Random', hypotheses, dm_strat='DC', axis=axis, exp_length=10, diss_length=10)

Max_step = 12000

for i in range(Max_step):
    a.random_walk()
    if i % 10 == 0:
        a.dm_object.make_decision(a.coo_array)
    a.plot_arena(i)