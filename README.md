# ES7-Project

This is the code repo for 6th semester project for Electronics Systems

## Modules

Each module has its own directory where it is as selfcontained as possible.
It is mixed wheter it is a bare virtual enviorment or uv mangaded.
If it is uv managed follow the instructions in specific `README.md` and ignore rest here. 

First `cd` to the desired module and follow directions in README.md

### Init uv enviorment
- If not already install uv: `https://docs.astral.sh/uv/getting-started/installation/`
- init: `uv init --bare -p <python-version>` e.g. `uv init --bare -p 3.13`
- add pip packages: `uv add <package list>`

First `cd` to the desired module and wollow directions in README.md

### Python modules

If a folder contains Python code, a virtual enviorment must be used within the folder named '.venv'

0. Fix execution policy in Powershell

   - Windows: `Set-ExecutionPolicy unrestricted`

1. Intall virtial enviorment (if not alredy):

   - Windows: `py -m pip install virtualenv`
   - Linux: `pip install virtualenv` _OR_ `sudo apt install python3.12-venv`

2. If not already created, create the virtual enviorment:

   - Windows: `py -m venv .venv`
   - Linux: `python3 -m venv .venv`

3. Activate the virtual enviorment:

   - Windows: `.venv\Scripts\activate`
   - Linux: `source .venv/bin/activate`

4. Install packages listed in associated README.md

   - Windows: `py -m pip install <modules>`
   - Linux: `pip install <modules>`

5. To deactivate the enviorment (after use):
   - Windows: `deactivate`
   - Linux: `source .venv/bin/deactivate`

## Training data location
- Simulation **without** noise: `ssh ubuntu@130.225.39.235:~/ESD7-Project/training_data_generation/sim_output/`
- Simulation **with** noise: `ssh ubuntu@130.225.39.212:~/ESD7-Project/training_data_generation/sim_output/`