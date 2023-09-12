"""
Perform an equation of state fit to the energy-volume data.
"""

import os
import re

from ase.io import read, write
import ase.spacegroup
import numpy as np
import json

class Prepare(object):
    """
    Prepare isotropically scaled POSCARs for an equation of state fit to the energy-volume data.
    """
    def __init__(self):
        pass


    def prepare_isotropic(self, poscar='POSCAR', points=21):
        """
        Prepare the isotropic volume scan for the equation of state fit.
        """
        
        poscar = read(poscar)
        scale_factors = np.linspace(0.90, 1.10, points) # recommended no larger than +/- 10%
        
        for iter, scale_factor in enumerate(scale_factors):
            poscar_scaled = poscar.copy()        
            poscar_scaled.set_cell(self.scale_volume(poscar_scaled.cell, scale_factor), scale_atoms=True)
            write(f'POSCAR.{iter}', poscar_scaled, format='vasp', vasp5=True)
        

    def scale_volume(self, cell, scale_factor):
        """
        Scale the volume of a generic cell shape by a fixed amount.
        """
        
        det = np.linalg.det(cell)
        scale = (scale_factor**(1/3))
        new_det = det * scale**3
        new_cell = cell * scale
        return new_cell
    

    def prepare_incar(self, poscar, incar='INCAR', method='gga'):
        """
        Prepare an INCAR file for the equation of state fit, reading options from incar_src/incar_options.json
        """

        import psutil
        import yaml

        VCPUS = psutil.cpu_count(logical=False)
        if VCPUS <= 48:
            NCORE = 4
        elif VCPUS > 48 and VCPUS < 128:
            NCORE = 8
        elif VCPUS >= 128:
            NCORE = 16
        
        poscar = read(poscar)

        incar_sources = 'incar_src/'
        incar_options = json.load(open(incar_sources + f'incar_eos_{method}.json', 'r'))

        magmom_yaml = yaml.load(open(incar_sources + 'magmoms.yaml', 'r'), Loader=yaml.FullLoader)
        magmoms = []
        for atom in poscar:
            try:
                magmoms.append(magmom_yaml['MAGMOM'][atom.symbol])
            except KeyError:
                magmoms.append(0.6)
        magmoms = ' '.join([str(m) for m in magmoms])

        with open(incar, 'w') as f:
            f.write(f'SYSTEM = Isotropic Equation of State\n\n')
            for key, value in incar_options.items():
                f.write(f'{key} = {value}\n')
            f.write(f'MAGMOM = {str(magmoms)}\n')
            f.write(f'NCORE = {NCORE}\n')


    def prepare_kpoints(self, poscar='POSCAR', kpoints='KPOINTS'):
        """
        Prepare a KPOINTS file for the equation of state fit.
        """

        poscar = read(poscar)
        
        # Calculate the reciprocal lattice vectors
        rec_lattice = poscar.get_reciprocal_cell()

        # Calculate the length of each reciprocal lattice vector
        rec_lengths = np.linalg.norm(rec_lattice, axis=1)

        # Calculate the k-point density based on the length of the shortest reciprocal lattice vector
        kppra = [int(np.ceil(np.pi / rl)) for rl in rec_lengths]

        # Write the KPOINTS file
        with open(kpoints, 'w') as f:
            f.write(f'Automatic mesh\n')
            f.write(f'0\n')
            f.write(f'Monkhorst-Pack\n')
            f.write(f'{kppra[0]} {kppra[1]} {kppra[2]}\n')
            f.write(f'0.0 0.0 0.0\n')


    def prepare_potcard(self, poscar='POSCAR', potcar='POTCAR'):
        """
        Prepare a POTCAR file for the equation of state fit.
        """

        poscar = read(poscar)

        potcar_sources = '/p/home/teep/.local_programs/vasp640/potpaw_PBE'
        potcar_symbols = set([atom.symbol for atom in poscar])
        

class Analyze(object):
    """
    Analyze the equation of state fit to the energy-volume data.
    """    
    def __init__(self):
        pass


    def analyze_isotropic(self, results_directory='vasp_out'):
        """
        Extract volume, final energy, and run time from the OUTCARs for the isotropic volume scan.
        """
        
        outcars = [f for f in os.listdir(results_directory) if f.startswith('OUTCAR.')]
        outcars.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
        
        energies = []
        volumes = []
        times = []
        
        for outcar in outcars:
            with open(os.path.join(results_directory, outcar), 'r') as f:
                lines = f.readlines()
                
                # Extract final energy
                for _, line in enumerate(lines):
                    if 'energy  without entropy' in line:
                        energy = float(line.split()[-2])
                        energies.append(energy)
                        break
                
                # Extract volume
                for _, line in enumerate(lines):
                    if 'volume of cell' in line:
                        volume = float(line.split()[-1])
                        volumes.append(volume)
                        break
                
                # Extract run time
                for _, line in enumerate(lines):
                    if 'Total CPU time used' in line:
                        time = float(line.split()[-2])
                        times.append(time)
                        break
        
        return energies, volumes, times


    def store_results(self, energies, volumes, times):
        """
        Store the results to a JSON file.
        """

        with open('results.json', 'w') as f:
            f.write(json.dumps({
                'energies': energies,
                'volumes': volumes,
                'times': times
            }, indent=4))


    def fit(self, energies, volumes, method='sj'):
        """
        Perform an equation of state fit to the energy-volume data using ASE.
        """

        from ase.eos import EquationOfState
        from ase.units import kJ

        eos = EquationOfState(volumes, energies, eos=method)
        v0, e0, B = eos.fit()
        eos.plot('eos.png')

        return v0, e0, B / kJ * 1.0e24


    def report_fitted_results(self, poscar, v0, e0, B):
        """
        Report the fitted results.
        """

        poscar = read(poscar)
        self.cell = poscar.cell

        # Using the reference POSCAR, scale the lattice vectors to match the fitted volume.
        poscar.set_cell(self.scale_volume(poscar.cell, v0 / np.linalg.det(poscar.cell)), scale_atoms=True)
        write(f'POSCAR.fitted', poscar, format='vasp', vasp5=True)
        
        print(f'v0 = {v0:.3f} Ang^3')
        print(f'lattice vectors = {poscar.cell} Ang')
        print(f'e0 = {e0:.3f} eV')
        print(f'B = {B:.3f} GPa')


class Workflow(object):
    """
    Schemas for the workflow.
    """

    def __init__(self):
        pass


    def prepare(self, poscar='POSCAR', points=21):
        """
        Prepare the isotropic volume scan for the equation of state fit.
        """

        Prepare().prepare_isotropic(poscar, points)
        Prepare().prepare_incar(poscar)
        Prepare().prepare_kpoints(poscar)

    def analyze(self, results_directory='vasp_out'):
        """
        Extract volume, final energy, and run time from the OUTCARs, perform fitting, store results as a JSON, and report the results to the terminal.
        """

        energies, volumes, times = Analyze().analyze_isotropic(results_directory)
        Analyze().store_results(energies, volumes, times)
        v0, e0, B = Analyze().fit(energies, volumes)
        Analyze().report_fitted_results('POSCAR', v0, e0, B)


# testing
#if __name__ == '__main__':
#    poscar = f"Test\n1\n4.2 1. 0.\n0. 4.2 1.\n0. 1. 4.2\nH\n1\nDirect\n0. 0. 0.\n"
#    with open('POSCAR', 'w') as file:
#        file.write(poscar)
#    prepare_isotropic('POSCAR', points=3)
