from I2EM import backscatter
import tqdm
import numpy as np
import matplotlib.pyplot as plt

fr = 39

eps = complex(3,0.01)

sig1 = 0.001  # rms height in m
L1 = 0.05  # correl length in m

sp = 'exponential'  # Exponential correlation function
xx = 3.5  # selection of correl func

angles = np.arange(5,65,5)

output = {'VV':[],
          'HH':[],
          'HV':[]}

for theta in tqdm.tqdm(angles):

    model_output = backscatter(fr,
                            sig1,
                            L1,
                            theta,
                            eps.conjugate(), # This is a total nightmare, fix this in later commit.
                            sp,
                            xx,
                            block_crosspol=True)

    output['VV'].append(model_output[0])
    output['HH'].append(model_output[1])
    output['HV'].append(model_output[2])

# print(output)

plt.plot(angles,output['VV'])
plt.plot(angles,output['VH'])
plt.show()
