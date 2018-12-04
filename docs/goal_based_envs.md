# Goal-based environments and `ObsDictRelabelingBuffer`
Some algorithms, like HER, are for goal-conditioned environments, like 
the [OpenAI Gym GoalEnv](https://blog.openai.com/ingredients-for-robotics-research/)
or the [multiworld MultitaskEnv](https://github.com/vitchyr/multiworld/) 
environments.

These environments are different from normal gym environments in that they 
return dictionaries for observations, like so: the environments work like this:

```
env = CarEnv()
obs = env.reset()
next_obs, reward, done, info = env.step(action)
print(obs)

# Output:
# {
#     'observation': ...,
#     'desired_goal': ...,
#     'achieved_goal': ...,
# }
```
The `GoalEnv` environments also have a function with signature
```
def compute_rewards (achieved_goal, desired_goal):
   # achieved_goal and desired_goal are vectors
```
while the `MultitaskEnv` has a signature like
```
def compute_rewards (observation, action, next_observation):
   # observation and next_observations are dictionaries
```
To learn more about these environments, check out the URLs above.
This means that normal RL algorithms won't even "type check" with these 
environments.

`ObsDictRelabelingBuffer` perform hindsight experience replay with 
either types of environments and works by saving specific values in the 
observation dictionary.

