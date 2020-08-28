import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

TIME_STEP = 0.01
TSPF = 10
#EXP_SCALE = 0.02
SAMPLE_NUM = 10


# COST_FACTOR = -1.

class epuck:
    def __init__(self, init_option):
        # constants
        self.vl = 0.16
        self.va = 0.75
        self.r = 0.035
        self.comm_dist = 0.5
        # movement
        self.dir = random.random() * 2 * math.pi
        self.dir_v = np.array([math.sin(self.dir), math.cos(self.dir)])
        self.turn_dir = int(random.random() * 2) * 2 - 1
        self.walk_state = int(random.random() * 2)
        self.walk_timer = 0
        # decision
        self.diss_state = 0
        self.option = init_option

class arena:
    def __init__(self, hypotheses, N=20, dim=np.array([2, 2]), axis=None):
        self.robots = []
        self.coo_array = np.array([]).reshape([0, 2])
        self.dir_v_array = np.array([]).reshape([0, 2])
        self.n = float(N)
        self.dim = dim
        self.global_gain = 0.
        self.local_gain_array = np.zeros(N)
        self.sum_dec = 0
        self.evaluation_num = 0
        for i in range(N):
            coo = np.array([random.random(), random.random()] * self.dim)
            while self.init_collision_detect(coo):
                coo = np.array([random.random(), random.random()] * self.dim)
                print('new position', i, coo)
            self.robots.append(epuck(init_option=np.random.choice(hypotheses)))
            self.coo_array = np.vstack((self.coo_array, coo))
        self.axis = axis

    def oob(self, coo):
        # out of bound
        if self.robots[0].r < coo[0] < self.dim[0] - self.robots[0].r \
                and self.robots[0].r < coo[1] < self.dim[1] - self.robots[0].r:
            return False
        else:
            return True

    def init_collision_detect(self, new_coo):
        # check if new_coo clip with any old coo, or oob
        if self.oob(new_coo):
            return True
        elif len(self.robots) == 0:
            return False
        else:
            dist_array = np.sqrt(np.sum((self.coo_array - new_coo) ** 2, axis=1))
            if np.min(dist_array) < 2 * self.robots[0].r:
                return True
            else:
                return False

    def collision_detect(self, n):
        if self.oob(self.robots[n].coo + self.robots[n].dir_v * 1 * self.robots[n].vl):
            return True
        else:
            return False

    def step_function(self, dir, x):
        if dir >= 0:
            return max(x, 0)
        else:
            return min(x, 0)

    def random_walk(self):
        for i in range(len(self.robots)):
            self.robots[i].walk_timer -= 1
            if self.robots[i].walk_state == 0:
                # going straight
                if (not self.collision_detect(i)) and self.robots[i].walk_timer > 0:
                    self.robots[i].coo += self.robots[i].dir_v * self.robots[i].vl * TIME_STEP
                    self.coo_array[i, :] = self.robots[i].coo
                else:
                    # start turning
                    self.robots[i].walk_state = 1
                    self.robots[i].walk_timer = random.random() * 4.5 / TIME_STEP
                    self.robots[i].turn_dir = int(random.random() * 2) * 2 - 1
            else:
                # turning
                if self.robots[i].walk_timer > 0:
                    self.robots[i].dir += self.robots[i].turn_dir * self.robots[i].va * TIME_STEP
                    self.robots[i].dir_v = np.array([math.sin(self.robots[i].dir), math.cos(self.robots[i].dir)])
                    self.dir_v_array[i, :] = self.robots[i].dir_v
                elif self.collision_detect(i):
                    self.robots[i].walk_timer = random.random() * 4.5 / TIME_STEP
                    self.robots[i].turn_dir = int(random.random() * 2) * 2 - 1
                else:
                    # start going straight
                    self.robots[i].walk_state = 0
                    self.robots[i].walk_timer = np.random.exponential(scale=40) / TIME_STEP
                    self.robots[i].dir_v = np.array([math.sin(self.robots[i].dir), math.cos(self.robots[i].dir)])
                    self.dir_v_array[i, :] = self.robots[i].dir_v

    def apply_dec_constraint(self, n):
        if n < 0.:
            return 0.
        elif n > 1.:
            return 1.
        else:
            return n

    def apply_dec_constraint_array(self, n):
        n_ = n
        n_[n < 0.] = 0.
        n_[n > 1.] = 1.
        return n_


    def plot_arena(self, t_step):
        if t_step % TSPF == 0:
            self.axis[0, 0].cla()
            self.axis[0, 1].cla()
            self.axis[1, 0].cla()
            self.axis[1, 1].cla()
            self.axis[0, 1].set_ylim([0, 1])

            self.axis[0, 0].plot(self.coo_array[:, 0], self.coo_array[:, 1], 'ro')
            self.axis[1, 0].plot(self.coo_array[:, 0], self.coo_array[:, 1], 'bo')
            for i in range(len(self.robots)):
                if self.robots[i].diss_state == 1:
                    self.axis[0, 0].plot(self.robots[i].coo[0], self.robots[i].coo[1], 'bo')
                self.axis[0, 0].plot([self.robots[i].coo[0], self.robots[i].coo[0] + self.robots[i].dir_v[0] * 0.1],
                                  [self.robots[i].coo[1], self.robots[i].coo[1] + self.robots[i].dir_v[1] * 0.1])
                self.axis[0, 1].plot(i, self.robots[i].option, 'r*')
            self.axis[0, 0].axis('equal')
            self.axis[0, 0].set(xlim=(0, self.dim[0]), ylim=(0, self.dim[1]))
            self.axis[1, 0].axis('equal')
            self.axis[1, 0].set(xlim=(0, self.dim[0]), ylim=(0, self.dim[1]))


            plt.draw()
            plt.pause(TIME_STEP * TSPF)
        else:
            pass
