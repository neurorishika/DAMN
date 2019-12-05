from subprocess import call

locusts = ['/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Locusts/Nov 2019 - IISER/Locust_A.locust','/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Locusts/Nov 2019 - IISER/Locust_B.locust']
protocols = ['/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odor Protocols/Nov 2019 - IISER/D1000.protocol','/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odor Protocols/Nov 2019 - IISER/D2000.protocol']
odors = ["/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/Nov 2019 - IISER/OdorA.odor","/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/Nov 2019 - IISER/OdorB.odor","/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/Nov 2019 - IISER/OdorC.odor"]
n_trials = 3

for l in range(n_trials):
        for j in protocols:
            for k in locusts:
                for i in odors:
                    print(i,j,k,l)
                    call(['python','initExperiment.py',i,j,k,str(l)])
