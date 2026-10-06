# Nanonets: Single-Electron Transport Simulator

High-performance Kinetic Monte Carlo (KMC) simulator for modeling single-electron transport in nanoparticle networks or any other small scale conductor network.

## 🚀 Installation (Local Development)

The recommended way to install and use `nanonets` locally is via a virtual environment. 

```bash
# 1. Clone the repository and navigate into it
git clone https://github.com/JonasMnsing/nanonets.git
cd nanonets

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the package
# Option A: Core simulator only (lightweight, recommended for HPC)
pip install -e .

# Option B: Full installation (includes Jupyter, Matplotlib, SciPy, and more for tutorials and advanced utilities)
pip install -e .[all]
```

## 🐳 Usage with Docker (Cross-Platform)

To avoid dependency conflicts or compiler issues (e.g., with Numba/LLVM on macOS or Windows), you can run your simulations in a fully isolated Docker container.

1. Build the Image

```bash
docker build -t nanonets:v1 .
```

2. Run a Custom Simulation Script
You can write your simulation script locally (e.g., `my_simulation.py`) and execute it inside the container. By mounting your current directory (`-v $(pwd):/app`), the generated data files will be saved directly to your host machine:

```bash
docker run -v $(pwd):/app nanonets:v1 python my_simulation.py
```

## 📖 Tutorials & Examples

We provide interactive Jupyter notebooks to help you get started with `nanonets`. These notebooks cover everything from basic setup to advanced simulations. For this, check out the [`tutorials/`](./tutorials/) directory.

*Note: If you want to run these notebooks locally, make sure to install Jupyter (`pip install jupyter` or add it to your environment).*