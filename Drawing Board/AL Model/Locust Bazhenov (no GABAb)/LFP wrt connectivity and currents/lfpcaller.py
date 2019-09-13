from subprocess import call

p = [0.33,0.67]
p_n = [3,6,9,12,15]
l_n = [3,6,9,12,15]

for i in p:
    for j in p_n:
        for k in l_n:
            call(['python','runLFPtest.py', str(i), str(j), str(k)])
