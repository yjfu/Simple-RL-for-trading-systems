import gym
import tensorflow as tf
import numpy as np
from collections import deque
import random
import myEnv

ENV_NAME = 'CartPole-v0'
GAMMA = 0.9
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.001
REPLAY_SIZE = 10000
BATCH_SIZE = 32
TEST=10
EPISODE = 10000
STEP = 300

class DQN():
    def __init__(self,env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape=shape)
        return tf.Variable(initial)
    def create_Q_network(self):
        w1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])
        w2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        self.state_input = tf.placeholder("float",[None,self.state_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input,w1)+b1)
        self.Q_value=tf.matmul(h_layer,w2)+b2
    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim])
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer)>REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer)>BATCH_SIZE:
            self.train_Q_network()
    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input:[state]})[0]
        self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/10000
        if random.random()<=self.epsilon:
            return random.randint(0,self.action_dim-1)
        else:
            return np.argmax(Q_value)

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
        })[0])
    def train_Q_network(self):
        self.time_step += 1
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i]+GAMMA*np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
        })

def main():
    env = myEnv.OneStock("dout",3,50)
    agent = DQN(env)

    for episode in xrange(EPISODE):
        state = env.reset(50)
        print "ep",episode,"\n"
        for step in xrange(STEP):
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break

        if (episode+1) % 50 == 0:
            fout = open("record","a")
            total_reward = 0
            for i in range(TEST):
                state = env.reset(0,50)

                fout.write(str(episode)+" of "+str(i)+":\n============================\n")
                for j in range(STEP):
                    #env.render()
                    action = agent.action(state)
                    close = env.y_[env.index-1]
                    state,reward,done,_=env.step(action)
                    close2 = env.y_[env.index-1]
                    total_reward += reward
                    fout.write(str(reward)+" with "+str((action-10)*10)+" from "+str(_[0])+"to"+str(_[1])+
                               "so we have "+str(state[0])+"\n")
                    if done:
                        break
                fout.write(str("=======================\ntotal reward:"+str(total_reward)+"\n=====================\n"))

            ave_reward = total_reward/TEST
            fout.write(str("**********************\nave_reward in this episode is:"+str(ave_reward)+"\n*******************\n"))
            fout.close()
            print 'episode: ',episode,'evaluation average reward:',ave_reward

            if ave_reward>20000:
                break
if __name__ == '__main__':
  main()
