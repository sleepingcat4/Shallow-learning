# Define the depth and width of each layer in the model
lenet5_depth = [1, 6, 1, 16, 1, 120, 1, 84, 1, 10]
lenet5_width = [32*32*3, 28*28*6, 14*14*6, 10*10*16, 5*5*16, 120, 84, 10]

import math
import numpy as np
from lenet5 import lenet5

# Calculate power law and conservation law for LeNet-5
model = lenet5()
lenet5_depth = [1, 6, 6, 16, 16, 120, 84, 10]
lenet5_width = [32*32*3, 28*28*6, 14*14*6, 10*10*16, 5*5*16, 120, 84, 10]

# Calculate power law
d1 = 1
d2 = lenet5_depth[-1]
w1 = lenet5_width[-2]
w2 = lenet5_width[-1]

constant = math.log(w2/w1)/math.log(d2/d1)
print("Power Law for LeNet-5: d2/d1 =", d2/d1, ", constant =", constant)

# Calculate complexity using the power law as decay error
epsilon = []
for i in range(len(lenet5_depth)):
    di = lenet5_depth[i]
    Ai = lenet5_width[i]
    rho = constant
    epsilon.append(Ai*(di**rho))
    
print("Complexity for LeNet-5 using power law as decay error:", epsilon)

# Calculate conservation law
conservation_constant = np.prod(np.array(lenet5_depth)*np.array(lenet5_width))**(1/len(lenet5_depth))
print("Conservation law for LeNet-5: depthi × mi =", conservation_constant)
