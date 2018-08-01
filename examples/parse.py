import sys
from socket import gethostname

def addhash(dic, key):
    if key not in dic:
        dic[key] = len(dic)
    return dic[key]

def prefix(fields):
    if fields[1] == "create":
        pfx1 = 'C'
    elif fields[1] == "exec":
        pfx1 = 'E'
    else:
        pfx1 = '?'
    if fields[2] == "convolution":
        pfx2 = 'C'
    elif fields[2] == 'reorder':
        pfx2 = 'R'
    elif fields[2] == 'pooling':
        pfx2 = 'P'
    elif fields[2] == 'eltwise':
        pfx2 = 'E'
    elif fields[2] == 'sum':
        pfx2 = 'S'
    else:
        pfx2 = '?'
    if 'backward' in fields[4]:
        pfx3 = 'B'
    elif 'forward' in fields[4]:
        pfx3 = 'F'
    elif 'undef' in fields[4]:
        pfx3 = 'U'
    else:
        pfx3 = '?'
    return pfx1 + pfx2 + pfx3
def parsefile(name):
    f = open(name, 'r')
    found = False
    dic = {}
    outname = 'mkldnn-hostname-' + gethostname() + '.csv'
    fout = open(outname, 'w')
    fout.write('name,start_tsc.RDTSC,end_tsc\n')
    for line in f:
        line = line.rstrip()
        if not found:
            if line == '===== start =====':
                found = True
            continue
        if line == '===== stop =====':
            break
        if not line.startswith('mkldnn_verbose'):
            continue
        fields = line.split(',')
        key = '+'.join(fields[1:8])
        slot = addhash(dic, key)
        pfx = prefix(fields)
        fout.write(pfx+str(slot) + ',' + str(fields[8]) + ',' + str(fields[9]) +
                '\n')
    f.close()
    fout.close()
    fout = open('key.csv', 'w')
    dict1 = {}
    for k in dic.keys():
        dict1[dic[k]] = k
    v = [(k,dict1[k]) for k in sorted(dict1.keys())]
    fout.write('\n'.join([str(v[k][0])+','+str(v[k][1]) for k in range(len(v))]))
    fout.close()


if __name__ == '__main__':
    parsefile(sys.argv[1])

