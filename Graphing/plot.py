import matplotlib.pyplot as plt

def plot(fileName, lim):
    with open (fileName, 'r') as file:
        line = file.readline()
        corrupt, c = 0, 0 # is data corrupt? count points plotted
        while line != '' and c < lim:
            strY = line.split('\t') # needs converstion string -> float (see for loop below)
            line = file.readline() # style comment: logical placement, but not intuitive
            X, Y, error = [i for i in range(len(strY))], [], 0  # error = 1 means corrupt data
            for item in strY:
                try:
                    Y.append(float(item))
                except ValueError:
                    corrupt += 1
                    error = 1
                    continue
            if error == 0:  # if no corrupt data
                plt.plot(X, Y)
                c += 1  # one more data set plotted ... Note: corrupt data is skipped
            continue   
    print('👉👉👉', c, 'data sets plotted with', corrupt, 'corrupt data skipped ⛔️⛔️⛔️\n')
    return 0 
