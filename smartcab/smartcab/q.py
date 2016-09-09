import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.discount_factor = 0.5
        self.learning_rate = 0.5
        self.default_Q = 2
        self.none_count = 0
        self.max_right = 4
        self.max_none = 6
        self.state0 = None
        self.action0 = None
        self.reward0 = None
        self.Q = {}
        self.trials = -1
        self.max_trials = 100
        self.x_trials = range(0,self.max_trials)
        self.y_trials = range(0,self.max_trials)
        for i in ['forward','left','right']:
            for j in ['green','red']:
                for k in self.env.valid_actions:
                    self.Q[(i,j),k] = self.default_Q
    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if (self.state0, self.action0, self.reward0) != (None, None, None):
            oldQ = self.Q[(self.state0,self.action0)]
            self.Q[(self.state0,self.action0)] = oldQ + self.learning_rate*(self.reward0 - oldQ)
        self.none_count = 0
        self.trials = self.trials + 1
        (self.state0, self.action0, self.reward0) = (None, None, None)
        
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'])
        # TODO: Select action according to your policy
        newQ = -9999999999
        action = None
        
        if self.Q[(self.state,None)] == self.Q[(self.state,'left')] == self.Q[(self.state,'right')] == self.Q[(self.state,'forward')]:
            action = random.choice(Environment.valid_actions)
            newQ = self.Q[(self.state,'left')]
        else:      
            for i2 in [None,'left','right','forward']:
                if self.Q[(self.state,i2)] >= newQ:
                    newQ = self.Q[(self.state,i2)]
                    action = i2
        if action == None:
            self.none_count = self.none_count + 1
        else:
            self.none_count = 0
        if self.none_count > self.max_none:
            action = random.choice(Environment.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)


        # TODO: Learn policy based on state, action, reward
        if (self.state0, self.action0, self.reward0) != (None, None, None):
            oldQ = self.Q[(self.state0,self.action0)]
            self.Q[(self.state0,self.action0)] = (1-self.learning_rate)*oldQ + self.learning_rate*(self.reward0 + self.discount_factor*newQ)
        (self.state0, self.action0, self.reward0) = (self.state, action, reward)
        if (deadline == 0) & (reward < 10):
            self.y_trials[self.trials] = 0
        else:
            self.y_trials[self.trials] = 1
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    # sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim = Simulator(e, update_delay=0.01)
    sim.run(n_trials=a.max_trials)  # press Esc or close pygame window to quit
    import pylab as pl
    pl.figure()
    pl.scatter(a.x_trials,a.y_trials)
    pl.legend()
    pl.xlabel('Trial #')
    pl.ylabel('Success = 1, Failure = 0')
    pl.title("Training progress report")
    pl.show()


if __name__ == '__main__':
    run()