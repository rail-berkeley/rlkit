from d4rl.kitchen.kitchen_envs import KitchenMultitaskAllV0

env = KitchenMultitaskAllV0(
    dense=False,
    fixed_schema=True,
    use_combined_action_space=False,
)
print(env.action_space.low)
print(env.action_space.high)
tasks = [
    "microwave",
    "top left burner",
    "light switch",
    "hinge cabinet",
    "slide cabinet",
]
env.reset()
d = True
ctr = 0
r = 0
for i in range(1):
    print("Hello")
    if d:
        print(r)
        print()
        env.reset()
        print(env.tasks_to_complete[0])

    o, r, d, i = env.step(
        env.action_space.sample(), render_every_step=True, render_mode="rgb_array"
    )
    ctr += 1
