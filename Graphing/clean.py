# rawFile, newFile = 'pearson.csv', 'pearson.txt'
rawFile, newFile = 'Graphing/data.txt', 'Model/data.txt'

def clean():
    with open (rawFile, 'r') as raw, open (newFile, 'w') as new:
        line = raw.readline()
        while line != '':
            newLine = line.split('\t', 1)[1]    # for pearson.csv replace \t by ,
            new.write(newLine)
            line = raw.readline()
    return 0

clean()