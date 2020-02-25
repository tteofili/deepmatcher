import matplotlib.pyplot as plt
from itertools import islice 

# Ti restituisce i primi n valori di datatasetOriginal
# con n = len(datasetOriginal) * percentuale
def splitting_dataSet(percentuale, dataSetOriginal):
    lunghezza=int(len(dataSetOriginal)*percentuale)
    "Return first n items of the iterable as a list"
    output=list(islice(dataSetOriginal, lunghezza))
     
    #print("Split length list: ", percentuale) 
    #print("List after splitting", output)
    return output

def plotting(result_list,index):
    sim_list=[]
    label_list=[]
    t=[]
    
    sim_listArr=[]
    
    sorted_by_secondEthird = sorted(result_list, key=lambda tup: (tup[3], tup[2][index]))
    for i in range(len(sorted_by_secondEthird)):
        
        sim_list.append(sorted_by_secondEthird[i][2][index])
        label_list.append(sorted_by_secondEthird[i][3])
        t.append(i)
        
    index_min=label_list.index(1)
    min_sim_match=sim_list[index_min]
    max_sim_noMatch=sim_list[index_min-1]
    average=(sum(sim_list) / len(sim_list))
    print(average)
    wrong_match=0
    wrong_NOmatch=0
    for i in range(len(sim_list)):
        
        if sim_list[i]>=average:
            if label_list[i]!=1:
                wrong_match=wrong_match+1
            sim_listArr.append(1)
        else:
            sim_listArr.append(0)
            if label_list[i]!=0:
                wrong_NOmatch=wrong_NOmatch+1
    
    plt.plot(t, label_list, '-b',t, sim_list, '-r')
    plt.ylabel('plot_sim'+str(index))
    plt.show()
    '''
    plt.plot(t, label_list, '-b',t, sim_listArr, '-r' )
    plt.ylabel('plot_simArr'+str(index))
    plt.show()
    
    print("wrong_match. "+str(wrong_match)) 
    print("wrong_NOmatch. "+str(wrong_NOmatch))    
    print(min_sim_match)'''
    return min_sim_match,max_sim_noMatch
    #return sim_list, label_list, t, sim_listArr
    
    
def plot_graph(result_list):
   
    for j in range(len(result_list[0][2])):
        print(j)
#        result_listANHAI1=ratio_dupl_noDup4Anhai(result_list,j)
#        shuffle(result_listANHAI1)
        dataset5Percent=splitting_dataSet(0.05, result_list)
        g=0
        k=0
        for i in range(len(dataset5Percent)):
            if dataset5Percent[i][3]==1:
                g=g+1
            else:
                k=k+1
        
        print("match number: "+str(g)+ " no match number: " + str(k))
        min_sim_match,max_sim_noMatch=plotting(dataset5Percent,j)
    return min_sim_match,max_sim_noMatch
def plot_pretrain(data):
    random_tuples1 = data[:1000]
    random_tuples2 = data[-1000:]


    random_tuples1 +=random_tuples2
    result_list=[]
    result_list = sorted(data, key=lambda tup: (tup[2][0]))
    
    
    sim_list=[]
    t=[]
    for i in range(len(result_list)):
        
        sim_list.append(result_list[i][2][0])
        t.append(i)
    plt.plot(t, sim_list, '-r')
    plt.ylabel('plot_pretraining data')
    plt.show()
    
def plot_dataPT(data):
    
    result_list=[]
    result_list = sorted(data, key=lambda tup: (tup[2][0]))
    
    
    sim_list=[]
    t=[]
    for i in range(len(result_list)):
        
        sim_list.append(result_list[i][2][0])
        t.append(i)
    plt.plot(t, sim_list, '-r')
    plt.ylabel('plot_pretraining dataset')
    plt.show()