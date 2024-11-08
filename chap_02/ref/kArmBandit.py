import numpy as np
from scipy.special import softmax


class KArmBandit:

    def __init__(
            self,
            k,
            stationary=True,
            eps=0,
            step_size=None,
            init_Q=0,
            ucb=False,  # upper confidence bound
            c=2,  # confidence level (# std)
            gradient=False,
            init_q=0,
            ) -> None:
        '''Initial bandit status
        k:  # arms
        q: real reward expectation
        Q: empirical reward expectation
        '''
        self.k = k
        self.a = np.array(range(k), dtype=int)  # action index
        self.stat = stationary
        self.eps = eps  # epsilon, the probability to do exploration in each time step
        self.step_size = step_size
        self.init_Q = init_Q
        self.ucb = ucb
        self.c = c
        self.grad = gradient
        self.init_q = init_q


    def reset(self):
        '''Reset the bandit condition for next simulation.
        Replace the responsibility of __init__ for the ease of simulation.
        Since we have to reset the condition between each run of simulation.
        '''
        # q, true expected value
        if self.stat:
            self.q = np.random.normal(loc=self.init_q, scale=1, size=self.k)  # reward distribution, mean of normal
        else:
            # non-stationary
            self.q = np.zeros(self.k)  # special case of initialization

        self.Q = np.zeros(self.k) + self.init_Q  # optimistic initial Q
        self.NA = np.zeros(self.k)  # # action visit
        self.t = 0


    def optimal_action(self):
        '''Get the optimal action (index).
        Reset the object before calling this method'''
        return np.argmax(self.q)


    def _q_update(self):
        if not self.stat:
            self.q += np.random.normal(loc=0, scale=0.01, size=len(self.q))


    def _act(self) -> tuple[int, float]:
        '''action to maximize overall reward then update the Q
        Just run 1 time'''

        def get_A_from_Qs(Qs):
            # See Q(a) for all a to decide the action (which arm to pull)
            max_Q = max(Qs)
            # Choose action
            A = [i for i, q in enumerate(Qs) if q == max_Q]  # argmax(Qs) but tolerent multiple choice
            A = A[np.random.randint(len(A))]  # if there are multiple action to choose, then randomly choose one
            return A


        self.t += 1  # time step

        # Choose an action, A
        if self.ucb:
            # + 1e-4 for validate the calculation when t = 1 (log(1) = 0)
            pseudo_Q = self.Q + self.c * np.sqrt(np.log(self.t + 1e-4) / self.NA)
            A = get_A_from_Qs(pseudo_Q)
        elif self.grad:
            prob = softmax(self.Q)
            A = np.random.choice(self.a, p=prob)
        else:
            # Exploration. Choose an action randomly
            if np.random.rand() < self.eps:  # exploration happen
                explore = 1  # for debug
                A = np.random.randint(self.k)
            else:
                A = get_A_from_Qs(self.Q)

        # sample reward
        R = np.random.normal(self.q[A], 1)

        # update decision value
        if self.grad:
            if self.t == 1:  # initial RpM
                self.RpM = R
            # action preference
            one_hot = np.zeros(self.k)
            one_hot[A] = 1
            self.Q += self.step_size * (R - self.RpM) * (one_hot - prob)
            # RpM(previous average reward)
            step_size = (1 / self.t) if self.stat else self.step_size
            self.RpM += step_size * (R - self.RpM)
        else:
            Q = self.Q[A]
            self.NA[A] += 1  # visit A once
            step_size = self.step_size if self.step_size else (1 / self.NA[A])
            self.Q[A] = Q + step_size * (R - Q)

        # update q for non-stationary condition
        self._q_update()
        return A, R


    def run(self, time):
        '''run several(time) times
        Return:
            1. reward each time
            2. is optimal action
        '''
        self.reset()
        res = np.zeros((2, time))
        for t in range(time):
            A, R = self._act()
            res[0, t] = R
            res[1, t] = A
        res[1, :] = (res[1, :] == self.optimal_action())
        return res


    def simulate(self, time=1000, run=2000):
        '''Make simulation to estimate the average results, average reward and
        optimal action %.
        time: # time steps
        run: # simulation runs
        '''
        res = np.array([self.run(time) for r in range(run)])
        return res.mean(axis=0)


if __name__ == "__main__":
    np.random.seed(0)

    # varify through pdb
    bandit = KArmBandit(4, eps=0.3)
    bandit.reset()
    print("q: {0}".format(bandit.q))
    print("Optimal action: {}".format(bandit.optimal_action()))
    # run each
    for i in range(10):
        bandit._act()

    # run in a roll
    reward, is_opt = bandit.run(1000)

    # non-stationary condition
    bandit = KArmBandit(4, stationary=False, eps=0.3)
    bandit.reset()
    print("q: {0}".format(bandit.q))
    for i in range(10):
        bandit._act()

    # constant step size
    bandit = KArmBandit(4, eps=0.3, step_size=0.1)
    bandit.reset()
    print("q: {0}".format(bandit.q))
    for i in range(10):
        bandit._act()

    # initial Q = 5
    bandit = KArmBandit(4, eps=0.3, step_size=0.1, init_Q=5)
    bandit.reset()
    print("q: {0}".format(bandit.q))
    for i in range(10):
        bandit._act()

    # simulation
    bandit = KArmBandit(4, eps=0.3)
    sim_res = bandit.simulate(time=300, run=100)

    # UCB
    bandit = KArmBandit(k=4, ucb=True)
    reward, is_opt = bandit.run(1000)

    bandit = KArmBandit(k=4, gradient=True, step_size=0.1)
    bandit.reset()
    for i in range(10):
        bandit._act()

