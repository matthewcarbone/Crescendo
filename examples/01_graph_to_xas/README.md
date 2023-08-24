# Graph to XAS predictions

This example tests Crescendo's ability to train graph neural networks to make predictions on X-ray spectroscopy data. The example data was constructed via the process:

```python
from crescendo.preprocess.materials_processing import construct_feff_dataset, interpolate_spectra

graphs, spectra = construct_feff_dataset("examples/01_graph_to_xas/data")
spectra = interpolate_spectra(spectra)
grid = spectra["grid"]
spectra = spectra["spectra"]

import pickle
pickle.dump(graphs[:6], open("examples/01_graph_to_xas/data/X_train.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(graphs[6:8], open("examples/01_graph_to_xas/data/X_val.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(graphs[8:], open("examples/01_graph_to_xas/data/X_test.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.HIGHEST_PROTOCOL
# 5

import numpy as np
np.savetxt("examples/01_graph_to_xas/data/Y_train.npy", spectra[:6, :])
np.savetxt("examples/01_graph_to_xas/data/Y_val.npy", spectra[6:8, :])
np.savetxt("examples/01_graph_to_xas/data/Y_test.npy", spectra[8:, :])
```
