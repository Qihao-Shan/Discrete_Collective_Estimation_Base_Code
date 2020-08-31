import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import comb


TIME_STEP = 0.01
TSPF = 20
#EXP_SCALE = 0.02
SAMPLE_NUM = 10


# COST_FACTOR = -1.

class epuck:
    def __init__(self):
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


class DM_object_DC:
    def __init__(self, tile_array, hypotheses, exp_length, diss_length, N=20):
        self.tile_array = tile_array
        self.decisions = np.random.choice(range(hypotheses.size), N)
        self.hypotheses = hypotheses
        #print('initialise decisions ', self.decisions)
        self.diss_state_array = np.zeros(N)
        self.n = N
        self.exp_length = exp_length
        self.diss_length = diss_length
        self.diss_timer = np.random.exponential(scale=self.exp_length)
        self.observation_array = np.zeros((N, 2))
        self.quality_array = np.zeros(N)

    def make_decision(self, coo_array):
        for i in range(self.n):
            if self.diss_state_array[i] == 0:
                # exploration
                colour = self.tile_array[int(coo_array[i, 0]), int(coo_array[i, 1])]
                observation = np.array([colour, 1 - colour])
                self.observation_array[i, :] += observation
                p = self.hypotheses[self.decisions]
                N = self.observation_array[i, 0] + self.observation_array[i, 1]
                k = self.observation_array[i, 0]
                self.quality_array[i] = comb(N, k) * p ** k * (1 - p) ** (N - k)
            else:
                # dissemination
                pass
                # collect opinions
                # make decision


class arena:
    def __init__(self, fill_ratio, pattern, hypotheses, dm_strat, exp_length, diss_length, N=20, dim=np.array([2, 2]), axis=None):
        # initialise arena
        self.length = int(dim[0]/0.1)
        self.width = int(dim[1]/0.1)
        self.tile_array = self.generate_pattern(self.length, self.width, fill_ratio, pattern)
        # initialise agents
        self.robots = []
        self.coo_array = np.array([]).reshape([0, 2])
        self.n = float(N)
        self.dim = dim
        for i in range(N):
            coo = np.array([random.random(), random.random()] * self.dim)
            self.robots.append(epuck())
            while self.collision_detect(self.coo_array, coo):
                coo = np.array([random.random(), random.random()] * self.dim)
                #print('new position', i, coo)
            self.coo_array = np.vstack((self.coo_array, coo))
        self.axis = axis
        if dm_strat == 'DC':
            self.dm_object = DM_object_DC(self.tile_array, hypotheses, exp_length, diss_length, N)
        elif dm_strat == 'DMVD':
            pass
        elif dm_strat == 'DMMD':
            pass
        else:
            pass

    def generate_pattern(self, length, width, fill_ratio, pattern):
        if pattern == 'Block':
            pass
        else:
            # random
            tiles = np.zeros(width * length)
            tiles[:int(tiles.size * fill_ratio)] = 1
            tiles = np.random.permutation(tiles)
            tiles = tiles.reshape((length, width))
            return tiles

    def oob(self, coo):
        # out of bound
        if self.robots[0].r < coo[0] < self.dim[0] - self.robots[0].r \
                and self.robots[0].r < coo[1] < self.dim[1] - self.robots[0].r:
            return False
        else:
            #print('oob ', coo)
            return True

    def collision_detect(self, coo_array, new_coo):
        # check if new_coo clip with any old coo, or oob
        if self.oob(new_coo):
            return True
        elif len(self.robots) == 1:
            return False
        else:
            dist_array = np.sqrt(np.sum((coo_array - new_coo) ** 2, axis=1))
            if np.min(dist_array) < 2 * self.robots[0].r:
                #print(dist_array)
                #print('collision ')
                return True
            else:
                return False

    def random_walk(self):
        for i in range(len(self.robots)):
            self.robots[i].walk_timer -= 1
            new_coo = self.coo_array[i, :] + self.robots[i].dir_v * self.robots[
                i].vl * TIME_STEP * 10  # check collision in next 10 time steps
            coo_array_ = np.delete(self.coo_array, i, 0)
            if self.robots[i].walk_state == 0:
                # going straight
                if (not self.collision_detect(coo_array_, new_coo)) and self.robots[i].walk_timer > 0:
                    self.coo_array[i, :] += self.robots[i].dir_v * self.robots[i].vl * TIME_STEP
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
                elif self.collision_detect(coo_array_, new_coo):
                    self.robots[i].walk_timer = random.random() * 4.5 / TIME_STEP
                    self.robots[i].turn_dir = int(random.random() * 2) * 2 - 1
                else:
                    # start going straight
                    self.robots[i].walk_state = 0
                    self.robots[i].walk_timer = np.random.exponential(scale=40) / TIME_STEP
                    self.robots[i].dir_v = np.array([math.sin(self.robots[i].dir), math.cos(self.robots[i].dir)])

    def plot_arena(self, t_step):
        if t_step % TSPF == 0:
            self.axis.cla()
            #self.axis[1].cla()
            #self.axis[1].set_ylim([-1, 4])

            self.axis.set_title('timestep '+str(t_step))
            for i in range(self.width):
                for j in range(self.length):
                    if self.tile_array[i, j] == 1:
                        self.axis.fill_between([i*0.1, (i+1)*0.1], [j*0.1, j*0.1], [(j+1)*0.1, (j+1)*0.1], facecolor='k')

            for i in range(len(self.robots)):
                circle = plt.Circle((self.coo_array[i, 0], self.coo_array[i, 1]), self.robots[0].r, color='r', fill=False)
                self.axis.add_artist(circle)
                self.axis.plot(np.array([self.coo_array[i, 0], self.coo_array[i, 0]+self.robots[i].dir_v[0]*0.05]), np.array([self.coo_array[i, 1], self.coo_array[i, 1]+self.robots[i].dir_v[1]*0.05]),'b')
            self.axis.plot(self.coo_array[:, 0], self.coo_array[:, 1], 'ro', markersize=3)
            #self.axis[1].plot(self.dm_object.decisions, 'r*')
            self.axis.set(xlim=(0, self.dim[0]), ylim=(0, self.dim[1]))
            self.axis.set_aspect('equal', adjustable='box')

            plt.draw()
            plt.pause(0.001)
        else:
            pass
