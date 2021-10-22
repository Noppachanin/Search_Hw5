#Type Search
from timeit import default_timer as timer
import math
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
def Linear(input, target):
    #print('Linear Search')
    #print("input target: %d"%target)
    start = timer()
    for i in range(len(input)):
        if input[i] == target:
            #print("Elapsed time : %.8f second(s)"%(timer()-start))
            return (timer()-start,i)
    return (timer()-start,-1)

def Binary(input, target):
    #print('Binary Search')
    #print("input target: %d"%target)
    start = timer()
    left = 0
    right = len(input) -1
    idx = -1
    while (left <= right) and (idx == -1):
        mid = (left+right)//2
        if input[mid] == target:
            idx = mid
        else:
            if target<input[mid]:
                right = mid -1
            else:
                left = mid +1

    #print("Elapsed time : %.8f second(s)"%(timer()-start))
    return (timer()-start,idx)



def Fibo(input, target):
    #print('Fibonacci Search')
    #print("input target: %d"%target)
    start = timer()
    # Define Fib = Fib1 + Fib2 where (Fib1, Fib2, Fib3) is a sequence
    Fib2 = 0
    Fib1 = 1
    Fib = Fib1 + Fib2
    while (Fib < len(input)):
        Fib2 = Fib1
        Fib1 = Fib
        Fib = Fib1 + Fib2
    index = -1
    while (Fib > 1):
        i = min(index + Fib2, (len(input)-1))
        if (input[i] < target):
            Fib = Fib1
            Fib1 = Fib2
            Fib2 = Fib - Fib1
            index = i
        elif (input[i] > target):
            Fib = Fib2
            Fib1 = Fib1 - Fib2
            Fib2 = Fib - Fib1
        else:
            #print("Elapsed time : %.8f second(s)"%(timer()-start))
            return (timer()-start,i)

    if(Fib1 and index < (len(input)-1) and input[index+1] == target):
        #print("Elapsed time : %.8f second(s)"%(timer()-start))
        return (timer()-start,index+1)
    #print("Elapsed time : %.8f second(s)"%(timer()-start))
    return (timer()-start,-1)

def RunExperiment1(num_loops=100,filename="Experiment1_search",size=[50,100,1000,5000],step=[1,5,10]):
    
    trial = []
    size_idx = 0
    for iter in range(12):
        step_idx = iter%3
        result_mid = [size[size_idx],step[step_idx]]

        input = np.arange(0,size[size_idx]*step[step_idx],step[step_idx])
        target = math.ceil(len(input)//2)
        sum_L, sum_B, sum_F = 0,0,0
        for rep in range(num_loops):
            L = Linear(input,input[target])
            sum_L += L[0]
            B = Binary(input,input[target])
            sum_B += B[0]
            F = Fibo(input,input[target])
            sum_F += F[0]

        if step_idx==2 :
            size_idx += 1

        result_mid.extend([sum_L*10**4, sum_B*10**4, sum_F*10**4])
        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Step size','Linear','Binary','Fibonacci'])
    df.to_csv(filename+".csv",index=False)
    print(df)

def RunExperiment1_sort(filename="Experiment1_search",size=[50,100,1000,5000],step=[1,5,10]):
    
    trial = []
    size_idx = 0
    for iter in range(12):
        step_idx = iter%3
        result_mid = [size[size_idx],step[step_idx]]

        input = np.arange(0,size[size_idx]*step[step_idx],step[step_idx])
        target = math.ceil(len(input)//2)
        
        
        L = Linear(input,input[target])
        
        B = Binary(input,input[target])
        
        F = Fibo(input,input[target])
        

        if step_idx==2 :
            size_idx += 1

        result_mid.extend([L[1], B[1], F[1]])
        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Step size','Linear','Binary','Fibonacci'])
    df.to_csv(filename+"_sorted.csv",index=False)
    print(df)

    trial = []
    size_idx = 0
    for iter in range(12):
        step_idx = iter%3
        result_mid = [size[size_idx],step[step_idx]]

        input = np.arange(0,size[size_idx]*step[step_idx],step[step_idx])
        target = math.ceil(len(input)//2)
        sum_L, sum_B, sum_F = 0,0,0
        np.random.shuffle(input)

        
        L = Linear(input,input[target])
        
        B = Binary(input,input[target])
        
        F = Fibo(input,input[target])
        

        if step_idx==2 :
            size_idx += 1

        result_mid.extend([L[1], B[1], F[1]])
        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Step size','Linear','Binary','Fibonacci'])
    df.to_csv(filename+"_unsorted.csv",index=False)
    print(df)


def RunExperiment2(num_loops=100,filename="Experiment2_search",size=[50,100,1000,5000]):
    
    trial = []
    for size_idx in range(len(size)):

        result_mid = [size[size_idx]]

        input = np.random.randint(500, size=size[size_idx])
        target = math.ceil(len(input)//2)

        sum_L, sum_B, sum_F = 0,0,0
        for rep in range(num_loops):
            L = Linear(input,input[target])
            sum_L += L[0]
            B = Binary(input,input[target])
            sum_B += B[0]
            F = Fibo(input,input[target])
            sum_F += F[0]

        result_mid.extend([sum_L*10**4, sum_B*10**4, sum_F*10**4])
        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Linear','Binary','Fibonacci'])
    df.to_csv(filename+".csv",index=False)
    print(df)

def RunExperiment3(num_loops=100,filename="Experiment3_search",size=[50,100,200],sd=[1,2.5,5]):
    
    trial = []
    size_idx = 0
    for iter in range(9):
        sd_idx = iter%3
        result_mid = [size[size_idx],sd[sd_idx]]

        input = np.random.normal(50, sd[sd_idx], size[size_idx])
        target = math.ceil(len(input)//2)
        sum_L, sum_B, sum_F = 0,0,0
        for rep in range(num_loops):
            L = Linear(input,input[target])
            sum_L += L[0]
            B = Binary(input,input[target])
            sum_B += B[0]
            F = Fibo(input,input[target])
            sum_F += F[0]

        if sd_idx==2 :
            size_idx += 1

        result_mid.extend([sum_L*10**4, sum_B*10**4, sum_F*10**4])
        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','SD','Linear','Binary','Fibonacci'])
    df.to_csv(filename+".csv",index=False)
    print(df) 

def RunExperiment4(num_loops=100,filename="Experiment4_search",size=[50,100,1000,5000],step=[1,5,10]):
    
    trial = []
    size_idx = 0
    

    for iter in range(12):
        step_idx = iter%3
        result_mid = [size[size_idx],step[step_idx]]
        input = np.arange(0,size[size_idx]*step[step_idx],step[step_idx])

        # Left target
        target = math.ceil(len(input)*0.1)
        sum_L, sum_B, sum_F = 0,0,0
        for rep in range(num_loops):
            L = Linear(input,input[target])
            sum_L += L[0]
            B = Binary(input,input[target])
            sum_B += B[0]
            F = Fibo(input,input[target])
            sum_F += F[0]
        result_mid.extend([sum_L*10**4, sum_B*10**4, sum_F*10**4])

        if step_idx==2 :
            size_idx += 1

        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Step size','Linear_p10','Binary_p10','Fibonacci_p10'])
    #df.to_csv(filename,index=False)
    df.to_csv(filename+"_10p.csv",index=False)
    print("----Table 1: target at 10%----")
    print(df)

    trial = []
    size_idx = 0
    for iter in range(12):
        step_idx = iter%3
        result_mid = [size[size_idx],step[step_idx]]
        input = np.arange(0,size[size_idx]*step[step_idx],step[step_idx])
        # Middle target
        target = math.ceil(len(input)*0.5)
        sum_L, sum_B, sum_F = 0,0,0
        for rep in range(num_loops):
            L = Linear(input,input[target])
            sum_L += L[0]
            B = Binary(input,input[target])
            sum_B += B[0]
            F = Fibo(input,input[target])
            sum_F += F[0]
        result_mid.extend([sum_L*10**4, sum_B*10**4, sum_F*10**4])

        if step_idx==2 :
            size_idx += 1

        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Step size','Linear_p50','Binary_p50','Fibonacci_p50'])
    #df.to_csv(filename,index=False)
    df.to_csv(filename+"_50p.csv",index=False)
    print("----Table 2: target at 50%----")
    print(df)
    
    trial = []
    size_idx = 0
    for iter in range(12):
        step_idx = iter%3
        result_mid = [size[size_idx],step[step_idx]]
        input = np.arange(0,size[size_idx]*step[step_idx],step[step_idx])
        
        # Right target
        target = math.ceil(len(input)*0.9)
        sum_L, sum_B, sum_F = 0,0,0
        for rep in range(num_loops):
            L = Linear(input,input[target])
            sum_L += L[0]
            B = Binary(input,input[target])
            sum_B += B[0]
            F = Fibo(input,input[target])
            sum_F += F[0]
        result_mid.extend([sum_L*10**4, sum_B*10**4, sum_F*10**4])

        if step_idx==2 :
            size_idx += 1

        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Step size','Linear_p90','Binary_p90','Fibonacci_p90'])
    #df.to_csv(filename,index=False)
    df.to_csv(filename+"_90p.csv",index=False)
    print("----Table 3: target at 90%----")
    print(df)

    trial = []
    size_idx = 0
    

    for iter in range(12):
        step_idx = iter%3
        result_mid = [size[size_idx],step[step_idx]]
        input = np.arange(0,size[size_idx]*step[step_idx],step[step_idx])

        # near first as target
        target = 1
        sum_L, sum_B, sum_F = 0,0,0
        for rep in range(num_loops):
            L = Linear(input,input[target])
            sum_L += L[0]
            B = Binary(input,input[target])
            sum_B += B[0]
            F = Fibo(input,input[target])
            sum_F += F[0]
        result_mid.extend([sum_L*10**4, sum_B*10**4, sum_F*10**4])

        if step_idx==2 :
            size_idx += 1

        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Step size','Linear_second','Binary_second','Fibonacci_second'])
    #df.to_csv(filename,index=False)
    df.to_csv(filename+"_second.csv",index=False)
    print("----Table 4: target at near the first element----")
    print(df)

    trial = []
    size_idx = 0
    

    for iter in range(12):
        step_idx = iter%3
        result_mid = [size[size_idx],step[step_idx]]
        input = np.arange(0,size[size_idx]*step[step_idx],step[step_idx])

        # last as target
        target = len(input)-1
        sum_L, sum_B, sum_F = 0,0,0
        for rep in range(num_loops):
            L = Linear(input,input[target])
            sum_L += L[0]
            B = Binary(input,input[target])
            sum_B += B[0]
            F = Fibo(input,input[target])
            sum_F += F[0]
        result_mid.extend([sum_L*10**4, sum_B*10**4, sum_F*10**4])

        if step_idx==2 :
            size_idx += 1

        trial.append(result_mid)

    df = pd.DataFrame(data=trial,columns=['Size of a list','Step size','Linear_last','Binary_last','Fibonacci_last'])
    df.to_csv(filename+"_last.csv",index=False)
    print("----Table 5: target at the last element----")
    print(df)


def RunHypothesis(filename="Experiment_Hypothesis"):
    # Average from linear as a benchmark

    input = np.arange(0,500,5)
    target = len(input)-1
    
    Binary_list = [Binary(input,target)[0]*10**4 for rep in range(100)]
    Fibo_list = [Fibo(input,target)[0]*10**4 for rep in range(100)]

    #Binary_mean = np.mean(Binary_list)
    #Fibo_mean = np.mean(Fibo_list)

    print('H0- mean time for Binary search and Fibonacci search are equal ')
    print('H1- mean time for Binary search and Fibonacci search are not equal ')

    #Hypothesis test
    tset, pval = ttest_1samp(Binary_list, np.mean(Fibo_list))
    if pval<0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")

RunHypothesis()
   
#RunExperiment1()
#RunExperiment1_sort()
#RunExperiment2()
#RunExperiment3()
#RunExperiment4()

