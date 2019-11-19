from subprocess import call

locusts = ['/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Locusts/EndSem/Locust_A.locust','/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Locusts/EndSem/Locust_B.locust']
protocols = ['/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odor Protocols/EndSem/D1000.protocol','/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odor Protocols/EndSem/D2000.protocol']
odors = ["/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/EndSem/OdorA.odor","/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/EndSem/OdorB.odor","/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/EndSem/OdorC.odor"]
n_trials = 3

for l in range(n_trials):
        for j in protocols:
            for k in locusts:
                for i in odors:
                    print(i,j,k,l)
                    call(['python','initExperiment.py',i,j,k,str(l)])
