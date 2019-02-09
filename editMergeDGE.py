#IMPORTS
from collections import defaultdict
import numpy as np
import pandas as pd
import editdistance
import multiprocessing as mp
from time import time


#specify number of processors to parallelize across
n_threads = mp.cpu_count() - 1 # use all but one processor
#n_threads = 5                 # use arbitrary number of processors
print('using ' + str(n_threads) + ' threads')

#LOAD DATA
data = pd.read_csv('gatheredUMIs.txt.gz', sep = '\t')


##########################################################
#FUNCTION DEFINITIONS

def groupByOneDegreeOfSeparation(barcodeSet):
    
    ungrouped = barcodeSet.copy()
    grouped = []
    
    #get all UMIs one separated by one edit distance from the one being checked
    #getting rid of unconnected ones
    neighbors = {}
    for b1 in barcodeSet:
        temp = [b2 for b2 in barcodeSet if editdistance.eval(b1,b2) == 1]
        if len(temp) > 0:
            neighbors[b1] = temp
        else:
            ungrouped.remove(b1)
            grouped.append([b1])
    
    while len(ungrouped)>0:
        
        b1 = next(iter(ungrouped))
        newMembers = set(neighbors[b1])
        group = newMembers.copy()
        group.add(b1)

        while len(newMembers)>0:
            temp = set()
            for b2 in newMembers:
                temp |= set(neighbors[b2]).difference(group)
            group |= temp
            newMembers = temp.copy()
        grouped.append([x for x in group])
        ungrouped -= group
    
    return grouped

#FUNCTION DEFINITIONS
def calculateDGE(data):
    
    #Initialize DGE
    dge = pd.DataFrame(columns=data.Gene.unique().astype(str),index=data['Cell Barcode'].unique())
    # enforce str because fly genome has a gene called nan
    
    #Count UMIs for each barcode separately
    for b in dge.index.tolist():
        uniqueGenes = np.array(list(data.loc[data['Cell Barcode']==b,'Gene'].unique().astype(str)))
        genes = np.array(list(data.loc[data['Cell Barcode']==b,'Gene'].astype(str)))
        UMIs = np.array(list(data.loc[data['Cell Barcode']==b,'Molecular_Barcode']))
        
        #Count UMIs for each gene separately
        #note that this means that the same UMI could contribute to different genes
        for g in uniqueGenes:
            #put unique UMIs into a set
            umiSet = set(UMIs[genes == g])
            mergedSets = groupByOneDegreeOfSeparation(umiSet)
            dge.loc[b,g] = len(mergedSets)

    return dge

def calculateDGEParallelized(data,n_threads):
    
    #SPLIT DATA INTO CHUNKS
    splitData = []
    uniqueCellBarcodeList = data['Cell Barcode'].unique()
    for b in uniqueCellBarcodeList:
        splitData.append(data.loc[data['Cell Barcode']==b])
    chunk_size = int(len(uniqueCellBarcodeList) / (n_threads))  # "optimal" chunk size
    #chunk_size = 1000                       # arbitrary chunk size
    n_chunks = int(len(uniqueCellBarcodeList) / chunk_size)
    dataChunks = []
    for i in range(0, n_chunks):
        if (i < n_chunks - 1):
            dataChunks.append(pd.concat(splitData[i*chunk_size:(i+1)*chunk_size]))
        else:
            dataChunks.append(pd.concat(splitData[i*chunk_size:]))
    
    # create the multiprocessing worker pool
    workers = mp.Pool(processes = n_threads)
    
    # start the workers and collect results
    # without any of that fancy TQDM nonsense
    multi_threaded_result = []
    for result in workers.imap_unordered(calculateDGE, dataChunks, chunksize=1):
        multi_threaded_result.append(result)
    workers.close()
    workers.join()
    
    dge = pd.concat(multi_threaded_result)

    return dge

###################################################################


#MERGE CELL BARCODES THAT NEED TO BE MERGED
groupedBarcodes = groupByOneDegreeOfSeparation(set(data['Cell Barcode'].unique()))
for g in groupedBarcodes:
    if len(g)> 1:
        data['Cell Barcode'] = data['Cell Barcode'].replace(to_replace=g[1:],value=g[0])

#GATHER UMIs FOR MERGED BARCODES
start = time()
dge = calculateDGEParallelized(data,n_threads)
end = time()
print('took ' + str((end-start)/60) + ' minutes')

#WRITE DGE TO FILE
dge.transpose().fillna(value=0).astype(int).to_csv(path_or_buf='dgeOneEditMerged.txt.gz',sep = '\t',compression='gzip')
