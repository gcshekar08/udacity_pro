import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pylab as pl

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.moves = 0
        self.qDict = dict()
        # learning rate
        self.alpha = 1
        self.epsilon = 0 
        self.gamma = 0.05       
        self.state = None
        self.new_state = None
        self.reward = None
        self.cum_reward = 0
        self.possible_actions = Environment.valid_actions
        self.action = None
        self.x_trials = range(0, 100)
        self.y_trials = range(0,100)
        self.trials = -1
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = None
        self.moves = 0
        self.state = None
        self.new_state = None
        self.reward = 0
        self.cum_reward = 0
        self.action = None
        self.trials = self.trials + 1
        
    #Get Q Value.    
    def getQvalue(self, state, action):
        key = (state, action)
        return self.qDict.get(key, 5.0)
    
    #Get MAx Q Value.
    def getMaxQ(self, state):
        q = [self.getQvalue(state, a) for a in self.possible_actions]
        return max(q)
    
    #epsilon-greedy approach
    def getaction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            q = [self.getQvalue(state, a) for a in self.possible_actions]
            if q.count(max(q)) > 1: 
                best_actions = [i for i in range(len(self.possible_actions)) if q[i] == max(q)]                       
                index = random.choice(best_actions)
            else:
                index = q.index(max(q))
            action = self.possible_actions[index]
        return action

    def qlearning(self, state, action, nextState, reward):
        key = (state, action)
        if (key not in self.qDict):
            self.qDict[key] = 5.0
        else:
            self.qDict[key] = self.qDict[key] + self.alpha * (reward + 
                                                              self.gamma* self.getMaxQ(nextState) - 
                                                              self.qDict[key])

        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.new_state = inputs
        self.new_state['next_waypoint'] = self.next_waypoint
        self.new_state = tuple(sorted(self.new_state.items()))
        # TODO: Select action according to your policy
        action = self.getaction(self.new_state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        # update q value.
        if self.reward != None:
            self.qlearning(self.state, self.action, self.new_state, self.reward)
        # set the state to the new state
        self.action = action
        self.state = self.new_state
        self.reward = reward
        self.cum_reward = self.cum_reward + reward
        self.moves = self.moves + 1
        
        if (deadline == 0) & (reward < 10):
            self.y_trials[self.trials] = 0
        else:
            self.y_trials[self.trials] = 1
        # TODO: Learn policy based on state, action, reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    # sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim = Simulator(e, update_delay=0.001, display=True)
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    #Plot Success. 
    pl.figure()
    pl.scatter(a.x_trials,a.y_trials)
    pl.legend()
    pl.xlabel('alpha = 1, epsilon = 0, gamma = 0.05')
    pl.ylabel('Success = 1, Failure = 0')
    pl.title("Training progress report")
    pl.show()
    
if __name__ == '__main__':
    run()
