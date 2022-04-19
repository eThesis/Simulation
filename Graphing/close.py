import matplotlib.pyplot as plt

def close(fileName):
    msg = '👉👉👉 Enter \'x\' to exit, enter \'s\' to save, enter both to do both: '
    do = input(msg)
    c = 0
    while True:
        c += 1
        if all(l in do for l in 'xs'):  # redundant but just for fun
            plt.savefig(fileName + '.pdf') # best save as .pdf although .svg is possible too
            plt.close()
            print("saved + closed")
            return 0
        if 's' in do:
            plt.savefig(fileName + '.png', dpi = 1000)
            print(c % 2, 'Saved!')
            do = input(str(c % 2) + msg)
        if 'x' in do:
            plt.close()
            print("closed without save")
            return 0
        do = input(str(c % 2) + msg)