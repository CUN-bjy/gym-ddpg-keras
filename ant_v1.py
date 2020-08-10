import roboschool
import gym

env = gym.make('RoboschoolAnt-v1')

observation = env.reset()

for _ in range(1000):
  
    env.render()
    
    #your agent goes here
    action = env.action_space.sample() 
         
    observation, reward, done, info = env.step(action) 
   
        
    if done: 
      break;
            
env.close()
