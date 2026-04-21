# DP-LBF

## Method Overview

* **RAPPOR**: A client-side local differential privacy (LDP) mechanism based on randomized response. It encodes each record into a Bloom filter representation and applies bitwise probabilistic perturbation, where the noise level is calibrated according to the degree of record correlation.


* **EBFDP**: A query-level differential privacy mechanism that applies perturbation to query outputs before release, ensuring formal differential privacy guarantees for each returned result.

* **DPBFE**: A differentially private Bloom filter encoding scheme for sequential numerical data. It adds Laplace noise to the Bloom filter bit array and applies a thresholding-based reconstruction mechanism.

* **DPBloomFilter**: A differential privacy-enhanced Bloom filter construction that incorporates sensitivity-aware noise injection. It leverages correlated sensitivity analysis to allocate perturbation more effectively, aiming to optimize the privacy–utility trade-off under the differential privacy framework.

* **DP-LBF**: A Bloom filter-based mechanism that employs the exponential mechanism to adaptively select the decision threshold, together with an auxiliary backup Bloom filter structure to mitigate utility degradation while preserving differential privacy guarantees.

---

## Datasets

* **Phishing URL Dataset**
  Source: [https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
  A labeled dataset containing phishing and legitimate URLs, widely used for binary classification and cybersecurity-related machine learning tasks.

* **Credit Card Default Dataset**
  Source: [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
  A real-world dataset containing anonymized credit card clients’ demographic and payment history information, commonly used for default risk prediction and classification benchmarks.

