import numpy as np  # Import if you're working with arrays
import pandas as pd


df = pd.read_csv("AIBL.csv")
def compare_values(DXCURREN, DXNORM):
    if len(DXCURREN) != len(DXNORM):
        return False  # Not the same if lengths differ

    # Option 1: Using allclose for numerical tolerance
    if np.allclose(DXCURREN, DXNORM):
        return True

    # Option 2: Element-wise comparison 
    for i in range(len(DXCURREN)):
        if DXCURREN[i] != DXNORM[i]:
            return False

    return True  # All elements are the same

# Example usage:
data_curr = [1.23, 5.67, 2.0, 9.1]
data_norm = [1.23, 5.67, 2.0, 9.100001]  # Last element slightly different

result = compare_values(data_curr, data_norm)
print(result)  # Will print False or True based on the comparison method
