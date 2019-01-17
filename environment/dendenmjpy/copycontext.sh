cp /Users/richard/.mujoco/mjkey.txt .
rsync -av -I --progress /Users/richard/pulkit/gym_envs/new_envs/gym_fetch_stack_multiworld . --exclude .git
rsync -av --progress /Users/richard/pulkit/existing_codebases/doodad . --exclude .git
rsync -av --progress /Users/richard/pulkit/existing_codebases/rlkit . --exclude .git --exclude rlkit_venv --exclude data