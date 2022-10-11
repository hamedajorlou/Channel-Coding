import gym
import numpy as np
from gym import wrappers ### for saving the file

env=gym.make('CartPole-v0')  ### making the environment
best_length=0  ### average of best lengths, may be updated with running each game
episode_lengths=[]  ### average of number of runs after running each game
best_weights=np.zeros(4) ### may be updated with running each game, will be used in the final game

######################
###Searching Part###
######################
for i in range(100): ### 100 different set of weights.
    # TODO: define random a vector of random uniform weights with length of 4 and values between 1 and -1.

    weights = np.random.uniform(-1.0,1.0,4)

    length=[] ### a list for the length of runs for each game

    # We are going to play the game 100 times for each set of weights and check the average length of trajectories for
    # each one.

    for j in range(100):  ### 100 different run for set of
        observation = env.reset()
        done=False ### check when the game is finished
        cnt=0 ###length of the trajectory for each the game

        while not done:
            cnt+=1
            # TODO: take the actions based on the random weights you've defined before.
            # Hint: make your decisions by the sign of the inner product of the weights and the observation vector/.
            action = ((np.sign(np.dot(weights, observation)) + 1) / 2).astype(int)

            # TODO: extract environment's new parameters with function called "step".
            # Hint: Use env.step
            observation, reward, done, info = env.step(action)
            if done:
                break

        length.append(cnt)

    average_length=sum(length)/len(length)  ###calculate the average length


    # TODO: compare the average and best length and update the best length and best weights if it's necessary.

    if average_length>best_length:
        best_length = average_length
        best_weights = weights

    episode_lengths.append(average_length)

    ### print the best length every 10 games
    if i%10 ==0:
        print('The best length is: '+str(best_length))

##################
###final game###
##################
done=False
cnt=0
env=wrappers.Monitor(env,'cartpole',force=True)

observation=env.reset() ###reset the environment

### play the game with best weights

while not done:
    # TODO: take the actions via the best set of weights you observed.
    # Hint: The code is pretty similar to the previous ones.
    cnt+=1;
    action = ((np.sign(np.dot(weights,observation))+1)/2).astype(int)
    observation,reward,done,info=env.step(action)
### print number of runs needed for final game
print('game lasted '+str(cnt)+' moves')
