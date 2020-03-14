# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:27:08 2020

@author: Giulia
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:52:38 2020

@author: Giulia
"""
import csv
from datasketch import MinHash, MinHashLSH
from nltk import ngrams
from plot import plotting,plot_dataPT 

def concatenate_list_data(list):
    result= ''
    for element in list:
        result += ' '+str(element)
    return result

def create_data(tableL, tableR, indiciL,indiciR):
   table1 = csv.reader(open(tableL,encoding="utf8"), delimiter=',')
   table2 = csv.reader(open(tableR,encoding="utf8"), delimiter=',') 
   next(table1, None)
   next(table2, None)
    
   #convert to list for direct access
   tableLlist = list(table1)
   tableRlist = list(table2)
   
    
   result_list= []
   result_list1,dataL=sampling_table(tableLlist,indiciL)
   result_list.extend(result_list1)
   result_list2,dataR=sampling_table(tableRlist,indiciR)
   result_list.extend(result_list2)
   return result_list,dataL,dataR

def sampling_table(table_list,indici):
   result_list1=[]
   data=[]
   for j in range(len(table_list)):
       table_el=[]
       for i1 in indici:
           table_el.append(table_list[j][i1])
       data.append(table_el)
       stringa_el=concatenate_list_data(table_el)
       lista_di_stringhe=stringa_el.split()
       result_list1.append(lista_di_stringhe)
   return result_list1,data  
   


def minHash_LSH(data):
    # Create an MinHashLSH index optimized for Jaccard threshold 0.5,
    # that accepts MinHash objects with 128 permutations functions
    # Create LSH index
    lsh = MinHashLSH(threshold=0.65, num_perm=256)
    
    # Create MinHash objects
    minhashes = {}
    for c, i in enumerate(data):
      #c è l'indice, i è la tupla
      #print(i)
      minhash = MinHash(num_perm=256)
      for el in i:
          minhash.update(el.encode('utf8'))
#      for d in ngrams(i, 3):
#        minhash.update("".join(d).encode('utf-8'))
      lsh.insert(c, minhash)
      minhashes[c] = minhash
      #print(str(c)+" "+str(minhashes[c]))
      
    res_match=[]
    for i in range(len(minhashes.keys())):
      result = lsh.query(minhashes[i])
      
      if result not in res_match and len(result)==2:
          res_match.append(result)
          #print("Candidates with Jaccard similarity > 0.6 for input", i, ":", result)
    #print(res)
#    for i in range(len(res_match)):
#        print(data[res_match[i][0]])
#        print(data[res_match[i][1]])
    return res_match

def create_dataset_pt(res, dataL,dataR,sim_function=lambda x, y: [1, 1]):
#    indL=len(dataL)-1
#    indR=len(dataR)-1
    dataPT=[]
    for el in res:
       el1=find_el(el[0],dataL,dataR) 
       el2=find_el(el[1],dataL,dataR) 
       
       sim_vector=sim_function(el1,el2)
       dataPT.append((el1,el2,sim_vector))
    
    return dataPT    

def find_el(index,dataL,dataR):
    if index>=len(dataL):
        indR=index-len(dataL)
        data_el=dataR[indR]
    else:
        data_el=dataL[index]
    
    return data_el

def split_indici(indici):
    indiciL=[]
    indiciR=[]
    for i in range(len(indici)):
        indiciL.append(indici[i][0])
        indiciR.append(indici[i][1])
    return indiciL,indiciR

def minHash_lsh(tableL, tableR, indici, simf):
    indiciL,indiciR=split_indici(indici)
    data4hash,dataL,dataR=create_data(tableL, tableR, indiciL,indiciR)
    res=minHash_LSH(data4hash)
    dataset_pt=create_dataset_pt(res, dataL,dataR,simf)
    plot_dataPT(dataset_pt)
    print(dataset_pt[:10])
    
    return dataset_pt


# TEST AREA #
if __name__ == "__main__":
    from sim_function import sim_cos, sim4attrFZ,sim4attrFZ_norm,sim4attrFZ_norm2
    from plot import plotting,plot_dataPT    
    #tableL='beer_exp_data/exp_data/tableA.csv'
    #tableR='beer_exp_data/exp_data/tableB.csv'
    tableL='fodo_zaga/fodors.csv'
    tableR='fodo_zaga/zagats.csv'
    indici=[(1, 1), (2, 2), (3, 3),(5,5)]
#    indiciL=[1,2,3,5]
#    indiciR=[1,2,3,5]
    
    simf = lambda a, b: sim4attrFZ(a, b)
    
    dataset_pt=minHash_lsh(tableL, tableR, indici,simf)
    #tableL='walmart_amazon/walmart.csv'
    #tableR='walmart_amazon/amazonw.csv'
    #indiciL=[5,4,3,14,6]
    #indiciR=[9,5,3,4,11]
    
    #data4hash,dataL,dataR=create_data(tableL, tableR, indiciL,indiciR)
    #res=minHash_LSH(data4hash)
    #dataset_pt=create_dataset_pt(res, dataL,dataR,sim4attrFZ)
    #print(dataset_pt[10:])
    
    #data = ['minhash is a probabilistic data structure for estimating the similarity between datasets',
    #  'finhash dis fa frobabilistic fata ftructure for festimating the fimilarity fetween fatasets',
    #  'weights controls the relative importance between minizing false positive',
    #  'wfights cfntrols the rflative ifportance befween minizing fflse posftive',
    #  "arnie morton\\'s of chicago 435 s. la cienega blv. los angeles 310/246-1501 american 0", 
    #  "arnie morton\\'s of chicago 435 s. la cienega blvd. los angeles 310-246-1501 steakhouses 0"]
    plot_dataPT(dataset_pt)