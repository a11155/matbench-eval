import os
from collections.abc import Callable
from typing import Any, Literal
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key
from tqdm import tqdm
from matbench_discovery import timestamp
from matbench_discovery.data import DataFiles, as_dict_handler, ase_atoms_from_zip
from matbench_discovery.enums import Task
import sys
from multiprocessing import Pool, cpu_count

absolute_path = os.path.abspath("/data/andrii/sevenNet")
sys.path.append(absolute_path)
from sevenn.sevennet_calculator import SevenNetCalculator

def process_single(args):
    atoms, calc, filter_name, optimizer_name, max_steps, force_max = args
    
    mat_id = atoms.info[Key.mat_id]
    try:
        atoms.calc = calc
        
        filter_cls = {"frechet": FrechetCellFilter, "exp": ExpCellFilter}[filter_name]
        optim_cls = {"FIRE": FIRE, "LBFGS": LBFGS}[optimizer_name]
        
        if max_steps > 0:
            atoms = filter_cls(atoms)
            optimizer = optim_cls(atoms, logfile="/dev/null")
            optimizer.run(fmax=force_max, steps=max_steps)
            
        energy = atoms.get_potential_energy()
        relaxed_struct = AseAtomsAdaptor.get_structure(getattr(atoms, "atoms", atoms))
        return mat_id, {"structure": relaxed_struct, "energy": energy}
    except Exception as exc:
        print(f"Failed to relax {mat_id}: {exc!r}")
        return mat_id, None

def main():
    # Configuration
    model_name = "./checkpoint_best.pth"
    task_type = Task.IS2RE
    job_name = f"{model_name}-wbm-{task_type}"
    ase_optimizer = "FIRE"
    ase_filter = "frechet"
    max_steps = 500
    force_max = 0.05
    n_processes = 4
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", "debug")
    slurm_array_task_count = 32

    os.makedirs(out_dir := "./results", exist_ok=True)
    out_path = f"{out_dir}/{model_name}-14-11-2024.json.gz"

    # Load data
    zip_filename = "./2024-08-04-wbm-initial-atoms.extxyz.zip"
    atoms_list = ase_atoms_from_zip(zip_filename)
    atoms_list = atoms_list

    if slurm_array_job_id != "debug" and slurm_array_task_count > 1:
        atoms_list = np.array_split(atoms_list, slurm_array_task_count)[slurm_array_task_id - 1]

    # Prepare arguments for parallel processing
    calc = SevenNetCalculator(model=model_name)
    process_args = [(atoms, calc, ase_filter, ase_optimizer, max_steps, force_max) 
                   for atoms in atoms_list]

    # Process in parallel
    relax_results = {}
    with Pool(processes=n_processes) as pool:
        for mat_id, result in tqdm(pool.imap_unordered(process_single, process_args), 
                                 total=len(process_args),
                                 desc="Relaxing structures"):
            if result is not None:
                relax_results[mat_id] = result

    # Save results
    df_out = pd.DataFrame(relax_results).T.add_prefix("sevennet_")
    df_out.index.name = Key.mat_id
    df_out.reset_index().to_json(out_path, default_handler=as_dict_handler)

if __name__ == "__main__":
    main()
