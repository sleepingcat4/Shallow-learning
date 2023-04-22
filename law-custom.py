import math
import numpy as np

# Calculate power law and conservation law for custom model
custom_depth = [1, 32, 32, 64, 64, 128, 128, 10]
custom_width = [32*32*3, 30*30*32, 28*28*32, 14*14*64, 12*12*64, 4*4*128, 2048, 10]

# Calculate power law
d1 = 1
d2 = custom_depth[-1]
w1 = custom_width[-2]
w2 = custom_width[-1]

constant = math.log(w2/w1)/math.log(d2/d1)
print("Power Law for custom model: d2/d1 =", d2/d1, ", constant =", constant)

# Calculate complexity using the power law as decay error
epsilon = []
for i in range(len(custom_depth)):
    di = custom_depth[i]
    Ai = custom_width[i]
    rho = constant
    epsilon.append(Ai*(di**rho))
    
print("Complexity for custom model using power law as decay error:", epsilon)

# Calculate conservation law
conservation_constant = np.prod(np.array(custom_depth[:-1])*np.array(custom_width[:-1]))**(1/len(custom_depth[:-1]))
print("Conservation law for custom model: depthi × mi =", conservation_constant)
