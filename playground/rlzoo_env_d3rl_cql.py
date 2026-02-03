import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams

env_name = "BreakoutNoFrameskip-v4"
env = create_test_env(
    env_name.gym_id,
    n_envs=args.n_envs,
    stats_path=maybe_stats_path,
    seed=args.seed,
    log_dir=log_dir,
    should_render=not args.no_render,
    hyperparams=hyperparams,
    env_kwargs=env_kwargs,
    vec_env_cls=ExperimentManager.default_vec_env_cls,
)