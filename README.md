# point-manipulation

A collection of general purpose functions for 2D/3D point manipulation

### Projective (homography) transformation

Example of estimating 2D projective (homography) transformation

```
import numpy as np
import matplotlib.pyplot as plt
import point_manipulation as pm

# Source rectangle points
src = np.array([[1,1.5],[2.5,1],[4,2.5],[2,3]])
# Destination rectangle points
dst = np.array([[0,0],[1,0],[1,1],[0,1]])

# Estimate a mapping from src to dst
T = pm.fit_homography(src,dst)
# Transform src points using the estimated mapping
srcT = pm.apply_transformation(src,T)

# Plot
plt.figure()
plt.plot(dst[:,0],dst[:,1],'bo-')
plt.plot(src[:,0],src[:,1],'ro-')
plt.plot(srcT[:,0],srcT[:,1],'m+-')
plt.axis('equal')
```