import sys

with open(sys.argv[1], 'r') as f:
    for line in f:
        vals = line.split(',')
        vals[112] = str(float(vals[112]) / 1031)
        line = vals[0]
        for i in range(1, 113):
            line += ',' + vals[i] 
        print(line)
