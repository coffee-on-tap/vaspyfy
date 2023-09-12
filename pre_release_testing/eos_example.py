import eos_fitting


def main():

    poscar = f"Test\n1\n4.2 1. 0.\n0. 4.2 1.\n0. 1. 4.2\nNi\n4\nDirect\n0. 0. 0.\n0. 0. 1.\n0. 0. 2.\n0. 0. 3.\n0. 0. 4.\n"
    with open('POSCAR', 'w') as file:
        file.write(poscar)
    p = eos_fitting.Workflow.prepare('POSCAR', points=3)

main()