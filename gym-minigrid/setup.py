from setuptools import setup

setup(
    name='gym_minigrid',
    version='1.0.2',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    description='Minimalistic gridworld package for OpenAI Gym',
    packages=['gym_minigrid', 'gym_minigrid.envs'],
    install_requires=[
        'gym==0.18.3'
    ]
)
