
import numpy as np
from gym import spaces
import tensorflow as tf
#return data,y_
def read_data(fileName, labelNum):
    rf = open(fileName, "r")
    rf.readline()
    rf.readline()
    buff = rf.readline()
    data = []
    y_ = []
    while buff:
        buff = buff.rstrip('\n')
        a = buff.split()
        b = []
        for i in range(1,labelNum):
            b.append(float(a[i]))
        for i in range(labelNum+1, a.__len__()):
            b.append(float(a[i]))
        y_.append(float(a[labelNum]))
        data.append(b)
        buff = rf.readline()
    rf.close()
    return data,y_

class OneStock():
    def __init__(self,fileName,labelNum,stop,start = 0):
        self.stop = stop
        self.fileName = fileName
        self.labelNum = labelNum
        self.data, self.y_ = read_data(fileName, labelNum)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0, -10, -100, 0, 0, 0, 0, 0, 0, 0]),
                                            np.array([100, 60, 60, 60, 500000, 10, 100,60, 60, 60, 400000, 400000, 400000, 30]))
        self.action_space = spaces.Discrete(21)
        if start == 0:
            self.index = self.data.__len__()-1
        else:
            self.index = start
        self.state = [10.0]+self.data[self.index]

        self.index -=1
    def step(self,action):
        newState = [self.state[0]]+self.data[self.index]

        newClose = self.y_[self.index]
        close = 0.0 if self.index == self.data.__len__() else self.y_[self.index+1]
        reward = 0.0

        if action < 10:
            reward = (newClose-close)*(newState[0]+(action-10)/10.0*newState[0])
            newState[0] = newState[0]+(action-10)/10.0*newState[0]

        else:
            reward = (newClose-close)*(newState[0]+(action-10)/10.0*(100-newState[0]))
            newState[0] = newState[0]+(action-10)/10.0*(100-newState[0])
        done = False
        if self.index == self.stop :
            done = True
        self.index -= 1
        self.state=newState
        return newState,reward,done,[close,newClose]
    def reset(self,stop,start = 0):
        self.__init__(self.fileName,self.labelNum,stop,start)
        return self.state
