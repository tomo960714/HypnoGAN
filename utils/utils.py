import numpy as np

def sin_noise(ns,seq_len,dim,offset = 0.0,amp = 1.0):
    """
    Sine wave generation.
    
    Args:
        ns (int): Number of sequences.
        seq_len (int): Length of each sequence.
        dim (int): Feature dimensions.
    
    Returns:
        data (list): The generated data.
    """
    
    # Initialize the output
    data = list()

    #Generate sin data
    for i in range(ns):
        #initialize time series:
        ts = list()
        #each feature
        for j in range(dim):
            #generate random frequency
            freq = np.random.uniform(0,0.1)
            #generate random phase
            phase = np.random.uniform(0,0.1)

            #adding random amplitude option
            if amp == 'random':
                #generate random amplitude
                amp = np.random.uniform(0.1,5)
            else:
                amp = float(amp)

            #adding random offset option
            if offset == 'random':        
                #generate random offset
                offset = np.random.uniform(-5,5)
            else:
                offset = float(offset)
            #generate time series
            ts_data= [offset + amp*np.sin(j*freq+phase) for j in range(seq_len)]
            ts.append(ts_data)

        #align row/column
        ts = np.transpose(np.asarray(ts))
        #normalize to [0,1]
        ts = (ts + 1) * 0.5
        #append to data
        data.append(ts)
    
    return data




