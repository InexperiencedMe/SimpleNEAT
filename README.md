# Core Diffusion
**SimpleNEAT** is an easy-to-study implementation of the legendary [NEAT Algorithm](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) from 2002, that belongs to a field of Machine Learning called Neuroevolution.

THUMBNAIL PICTURE HERE

This is an educational and exploratory repo, that prioritizes simplicity, code clarity and good abstractions. I made it as a fun project for my video (LINK NEEDED), that I made as an attempt to replicate what SethBling did in 2015 with Super Mario World.

## Showcase Training Runs

### Lunar Lander (Default Config)

PUT LUNAR LANDER GIF HERE AAAAAAAAAAAAAAAAAAAAAAAAAAAA

This result has been obtained in a matter of minutes. It could be done even faster if you make the parameters "more greedy", but I like to ensure sufficient exploration.

### Super Mario World (Default Config)

PUT SUPER MARIO WORLD GIF HERE AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

## Features
Currently, the project supports:
- **All original NEAT features (with modern improvements):** This repo remains faithful to the original idea - up to a certain point. Each feature has been implemented, and if it doesn't exist it's because I found it unnecessary and deleted it after some experiments. All the core mechanisms exist, but I also dynamically adjust the genetic compatibility thresholds, and do other things like create neurons with a synapse split, but I keep the old synapse active to not decrease the performance, even temporarily. That also makes `enabled` property of a Synapse not needed and I could completely delete it in the future. So, minor details like this might be different, but there was always a reason for the change. If you're looking for maximally faithful replication of the original, this is not it.

- **Deterministic Training:** Seed is specified in the config for the training to be fully reproducible. The only stochastic thing is visualization, that doesn't implement the seed.

- **Fully Modifiable Config:** Each run runs from a config, that specifies NEAT algorithm parameters, folders, run name, seed and visualization options. Each config is directly specified in the main script, but ideally it would be parsed as a script parameter to easily automate experiments.

- **Checkpoints:** Checkpoints are automatically saved, given that `saveCheckpoints` flag in the config is set to `True`. Checkpoints are saved everytime a new best fitnes score is achieved. Checkpoints are loaded by providing a `resumePath` argument to a `runEvolution` function. For example: `resumePath="checkpoints/superMarioWorldNEAT-16x16/Gen_194-Fitness_1617.pkl"`.

- **Visualizations:** Each neural net's operation can be visualized with all inputs (only up to 2D inputs), synapses (by default it's not static, but literally what state of activation they have during each frame), hidden neurons and outputs. Everything that I showed in my video is possible, but I am not happy with the abstractions. If I were seriously working on this project, I'd rewrite the whole visualization pipeline knowing what I know now.

- **Automatic Run Termination:** As a rule, runs terminate automatically when `targetFitness` specified in the config is achieved, but in practice I usually set the target to be astronomically high and just termine the run with a `Ctrl+C` when I'm satisfied.

- **Parallel Training:** Even though I don't prioritize speed and it's not a professional nor industrial repo, but for vast speedups I implemented multicore training, since we have to constantly evaluate the whole population, which is a highly parallelizable tasks. By default training utilizes all CPU cores, so don't be surprised if your computer runs slower. You can modify the number of workers in `trainer.py` file. Ideally, this could be modified in the config, but.. you know. My free time is limited.

- **Fully Connected Synapse Initialization:** If `initializeSynapses` flag is set to True in the config, the Synapses will be initialized to connect each input neuron to each output neuron. This tremendously speeds up the training, but for very complex environments with many inputs and/or outputs, it's a slight overkill.

## Usage

### Installation

To get started, clone the repository and install the project in editable mode so any changes you make to the code are applied immediately. 

```bash
git clone https://github.com/InexperiencedMe/SimpleNEAT.git
cd SimpleNEAT
pip install -e .
```

If you plan to run the included examples, you will also need the optional environment dependencies (`gymnasium` and `stable-retro`). You can install them alongside the package using:

```bash
pip install -e .[examples]
```

<br /> 

### Running Examples

There are ready-to-run scripts in the `examples/` folder so you can see the library in action and study how it works.

#### Lunar Lander
Requires `gymnasium` to be installed.
```bash
python examples/lunarLander.py
```

<br /> 

#### Super Mario World
Requires `stable-retro` to be installed. 
```bash
python examples/superMarioWorld.py
```
**Note on Super Mario World ROM:** Due to legal reasons, game ROMs are not included. You must legally acquire your own Super Mario World ROM and import it using the `stable-retro` tool. Downloading ROMs from sites like [https://romsfun.com/download/super-mario-world-7020](https://romsfun.com/download/super-mario-world-7020) is officially not recommended, but if it was legal, I would download the USA version and use it directly. Please follow the [stable-retro's documentation on importing ROMs](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms) before running this example.

Both examples are pre-configured. You can modify their respective YAML configuration files in the `configs/` folder to tweak all the parameters.

<br /> 

### Custom Environments

If you want to use SimpleNEAT for your own projects or custom environments, the best approach is to study the scripts in the `examples/` folder. 

The examples serve as a template. By reviewing them, you will figure out exactly what SimpleNEAT handles for you automatically, and what you need to provide yourself.

Please pay attention especially to my environment wrapper, since I mostly bypass the gymnasium API and create my own minimalistic API, that always bypasses the `info` variable, and combines termination and truncation into one `done` variable and stuff like this.

Generally speaking, if you just implement all features of the current examples, you'll be good to go. For Lunar Lander the example main script is fewer than 29 lines.
