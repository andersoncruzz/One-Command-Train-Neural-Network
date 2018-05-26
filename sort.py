def getValue(element):
    #print("element: ", element)
    aux = element.replace("weights.", "").split("-")
    return int(aux[0])

def insertionSort(vec):
    for index in range(1,len(vec)):
        currentvalue = vec[index]
        position = index

        while position>0 and getValue(vec[position-1]) > getValue(currentvalue):
            vec[position] = vec[position-1]
            position = position-1

        vec[position] = currentvalue

vec = ['weights.07-0.92.hdf5', 'weights.150-0.92.hdf5',
       'weights.122-0.92.hdf5','weights.207-0.92.hdf5',
       'weights.204-0.92.hdf5','weights.110-0.92.hdf5',
       'weights.334-0.98.hdf5', 'weights.320-0.85.hdf5',
       'weights.06-0.91.hdf5', 'weights.08-0.92.hdf5']

#alist = [54,26,93,17,77,31,44,55,20]

insertionSort(vec)
print(vec)
