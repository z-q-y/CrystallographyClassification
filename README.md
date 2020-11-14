# `AtomicGraphNets.jl` Crystallography Classification Example

The contents of this repo require two private Julia modules, `AtomicGraphNets.jl` and `ChemistryFeaturization.jl` to run. They are component packages of the ARPA-E project, and may be made available conditionally in the future. Other dependencies include:

- `Flux.jl`
- `GeometricFlux.jl`
- `PyCall.jl`, with access to the `pymatgen` and `rdkit` Python packages
- Other dependencies used by `AtomicGraphNets.jl` and `ChemistryFeaturization.jl` that may not be reflected here

Please run the Julia files using their eponymous Slurm batch scripts, so as not to take up precious computing resources on your head node.

## Setup instructions
1. Install the Julia dependencies outlined in `Manifest.toml`, and the Python dependencies mentioned above.
2. Download and untar the training data from crystallography.net, and place this repo under the `cod/` directory, preferably as `cod/src/`.
3. Generate training labels from the raw data by following instructions in `label_generator.ipynb`.
4. Preprocess the raw data into graphs that we can feed directly into `AtomicGraphNets.jl` models.
   - If access to multi-core mainframes (that use Slurm as their task managers) is available, type `sbatch graph_preprocessor_parallel.sh` in the terminal to run.
   - Elif access to machines that use Slurm as their task managers is available, type `sbatch graph_preprocessor.sh` in the terminal to run.
   - Run `graph_preprocessor.jl` otherwise.
5. Run any of the `train_func.jl` variants to train.
