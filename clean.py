rawFile, newFile = 'raw.txt', 'data.txt'

def clean():
    with open (rawFile, 'r') as raw, open (newFile, 'w') as new:
        line = raw.readline().strip().split()[1:]
        N = len(line)
        while N > 0:
            if N == 7:
                string = ''
                j = 0
                for i in range(len(line)):
                    if j < 6:
                        string += line[i] + '\t'
                        j += 1
                    else:
                        string += line[i]
                string += '\n'
                new.write(string)

                line = raw.readline().strip().split()[1:]
                N = len(line)
            else:
                line = raw.readline().strip().split()[1:]
                N = len(line)
                continue
    return 0

clean()