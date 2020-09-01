import numpy as np
import matplotlib.pyplot as plt
from arena_class import arena

fig, axis = plt.subplots(2,2)
hypotheses = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
resample_prob = 0.02
a = arena(0.4, 'Random', hypotheses, dm_strat='DC', axis=axis, exp_length=10, diss_length=10, resample_prob=resample_prob)

Max_step = 30000

for i in range(Max_step):
    a.random_walk()
    if i % 10 == 0:
        a.dm_object.make_decision(a.robots, a.coo_array)
    a.plot_arena(i)

print(a.dm_object.decision_array)