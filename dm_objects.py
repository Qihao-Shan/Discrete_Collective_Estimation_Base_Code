import numpy as np
import random


def check_dominant_decision(dm_object, threshold):
    decisions_tally = np.bincount(dm_object.decision_array)
    dominant_decision = np.argmax(decisions_tally)
    consensus_number = np.max(decisions_tally)
    if consensus_number > dm_object.n * threshold:
        return dominant_decision
    else:
        return -1


def normalise(q_array):
    return q_array/np.sum(q_array)


class DM_object_DC:
    def __init__(self, hypotheses, exp_length, diss_length, resample_prob, N=20):
        self.dm_type = 'dc'
        self.tile_array = np.array([])
        self.decision_array = np.random.choice(range(hypotheses.size), N)
        self.hypotheses = hypotheses
        self.hypotheses_mat = np.vstack((hypotheses, 1 - hypotheses))
        self.diss_state_array = np.zeros(N)
        self.n = N
        self.exp_length = exp_length
        self.diss_length = diss_length
        self.diss_timer_array = np.random.exponential(scale=self.exp_length, size=N)
        self.quality_array = np.zeros(N)
        self.quality_mat_self = np.ones((N, hypotheses.size))
        self.neighbour_mat = np.zeros((N, N))
        self.neighbour_decision_mat = -np.ones((N, N))
        self.neighbour_quality_mat = -100*np.ones((N, N))
        self.comm_dist = 0
        self.resample_prob = resample_prob

    def make_decision(self, robots, coo_array):
        self.diss_timer_array -= 1
        for i in range(self.n):
            if self.diss_state_array[i] == 0:
                # exploration
                if robots[i].walk_state == 0:
                    colour = self.tile_array[int(coo_array[i, 0]/0.1), int(coo_array[i, 1]/0.1)]
                    observation = np.array([colour, 1 - colour])
                    self.quality_mat_self[i, :] = normalise(self.quality_mat_self[i, :] *
                                                                 observation.dot(self.hypotheses_mat))
                self.quality_array[i] = self.quality_mat_self[i, self.decision_array[i]]
                if self.diss_timer_array[i] < 0:
                    self.diss_timer_array[i] = np.random.exponential(scale=self.diss_length)
                    self.diss_state_array[i] = 1
            else:
                # dissemination
                # collect opinions
                dist_array = np.sqrt(np.sum((coo_array - coo_array[i, :])**2, axis=1))
                self.neighbour_mat[i, dist_array <= self.comm_dist] = 1
                self.neighbour_decision_mat[i, dist_array <= self.comm_dist] = self.decision_array[dist_array <= self.comm_dist]
                self.neighbour_quality_mat[i, dist_array <= self.comm_dist] = self.quality_array[dist_array <= self.comm_dist]
                # make decision
                if self.diss_timer_array[i] < 0:
                    self.decision_array[i] = self.neighbour_decision_mat[i, np.argmax(self.neighbour_quality_mat[i, :])]
                    self.neighbour_mat[i, :] = np.zeros(self.n)
                    self.neighbour_decision_mat[i, :] = -np.ones(self.n)
                    self.neighbour_quality_mat[i, :] = -100*np.ones(self.n)
                    self.diss_timer_array[i] = np.random.exponential(scale=self.exp_length)
                    self.diss_state_array[i] = 0
                    r = random.random()
                    if r < self.resample_prob:
                        available = np.array([self.decision_array[i]-1, self.decision_array[i]+1])
                        available = available[available < self.hypotheses.size]
                        available = available[0 <= available]
                        self.decision_array[i] = np.random.choice(available)


class DM_object_individual:
    def __init__(self, hypotheses, N=20):
        self.dm_type = 'indi'
        self.tile_array = np.array([])
        self.decision_array = np.random.choice(range(hypotheses.size), N)
        self.hypotheses = hypotheses
        self.hypotheses_mat = np.vstack((hypotheses, 1 - hypotheses))
        self.n = N
        self.quality_mat_self = np.ones((N, hypotheses.size))
        self.diss_state_array = np.zeros(N)

    def make_decision(self, robots, coo_array):
        for i in range(self.n):
            if robots[i].walk_state == 0:
                colour = self.tile_array[int(coo_array[i, 0] / 0.1), int(coo_array[i, 1] / 0.1)]
                observation = np.array([colour, 1 - colour])
                self.quality_mat_self[i, :] = normalise(self.quality_mat_self[i, :] *
                                                        observation.dot(self.hypotheses_mat))
        s = self.quality_mat_self.sum(axis=1)
        d = np.argmax(self.quality_mat_self, axis=1)
        self.decision_array[s != self.hypotheses.size] = d[s != self.hypotheses.size]


class DM_object_DMVD:
    def __init__(self, hypotheses, exp_length, diss_length, resample_prob, N=20):
        self.dm_type = 'dmvd'
        self.tile_array = np.array([])
        self.decision_array = np.random.choice(range(hypotheses.size), N)
        self.hypotheses = hypotheses
        self.hypotheses_mat = np.vstack((hypotheses, 1 - hypotheses))
        # print(self.hypotheses_mat)
        # print('initialise decisions ', self.decisions)
        self.diss_state_array = np.zeros(N)
        self.n = N
        self.exp_length = exp_length
        self.diss_length = diss_length
        self.diss_timer_array = np.random.exponential(scale=self.exp_length, size=N)
        #self.observation_array = np.zeros((N, 2))
        self.quality_array = np.zeros(N)
        self.quality_mat_self = np.ones((N, hypotheses.size))
        self.neighbour_mat = np.zeros((N, N))
        self.neighbour_decision_mat = -np.ones((N, N))
        self.neighbour_quality_mat = -100 * np.ones((N, N))
        self.comm_dist = 0
        self.resample_prob = resample_prob

    def make_decision(self, robots, coo_array):
        self.diss_timer_array -= 1
        for i in range(self.n):
            if self.diss_state_array[i] == 0:
                # exploration
                if robots[i].walk_state == 0:
                    colour = self.tile_array[int(coo_array[i, 0]/0.1), int(coo_array[i, 1]/0.1)]
                    observation = np.array([colour, 1 - colour])
                    #self.observation_array[i, :] += observation
                    self.quality_mat_self[i, :] = normalise(self.quality_mat_self[i, :] *
                                                                 observation.dot(self.hypotheses_mat))
                self.quality_array[i] = self.quality_mat_self[i, self.decision_array[i]]
                if self.diss_timer_array[i] < 0:
                    self.diss_timer_array[i] = np.random.exponential(scale=self.diss_length * self.quality_array[i])
                    self.diss_state_array[i] = 1
            else:
                # dissemination
                # collect opinions
                dist_array = np.sqrt(np.sum((coo_array - coo_array[i, :])**2, axis=1))
                boolean_index = np.logical_and(dist_array <= self.comm_dist, self.diss_state_array == 1)
                self.neighbour_mat[i, boolean_index] = 1
                self.neighbour_decision_mat[i, boolean_index] = self.decision_array[boolean_index]
                self.neighbour_quality_mat[i, boolean_index] = self.quality_array[boolean_index]
                # make decision
                if self.diss_timer_array[i] < 0:
                    self.decision_array[i] = np.random.choice(self.neighbour_decision_mat[i, self.neighbour_mat[i, :] == 1])
                    self.neighbour_mat[i, :] = np.zeros(self.n)
                    self.neighbour_decision_mat[i, :] = -np.ones(self.n)
                    self.neighbour_quality_mat[i, :] = -100*np.ones(self.n)
                    self.diss_timer_array[i] = np.random.exponential(scale=self.exp_length)
                    self.diss_state_array[i] = 0
                    r = random.random()
                    if r < self.resample_prob:
                        available = np.array([self.decision_array[i]-1, self.decision_array[i]+1])
                        available = available[available < self.hypotheses.size]
                        available = available[0 <= available]
                        self.decision_array[i] = np.random.choice(available)
                        #print('resample ', i, self.decision_array[i])
