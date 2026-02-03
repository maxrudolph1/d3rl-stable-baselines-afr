import d3rlpy

# prepare dataset (1% dataset)
dataset, env = d3rlpy.datasets.get_atari_transitions(
    'breakout',
    fraction=0.01,
    num_stack=4,
)

# prepare algorithm
cql = d3rlpy.algos.DiscreteCQLConfig(
    observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
    reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
    compile_graph=True,
).create(device='cuda:0')

# start training
cql.fit(
    dataset,
    n_steps=1000000,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env, epsilon=0.001)},
)