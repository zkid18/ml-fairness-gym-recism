from gym.envs.registration import register

_env_specs = {
    "id": "MovieLensFairness-v0",
    "entry_point": "environments.recommenders.movie_lens_fairness_env:MovieLensFairness",
    "max_episode_steps": 50,
}
register(**_env_specs)