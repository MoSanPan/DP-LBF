# DLDP-BF
DP-LBF_2025

## File structure

* RAPPOR: A randomized response-based approach that introduces noise proportional to the number of correlated records. It enables client-side local differential privacy by perturbing each bit in a Bloom filter representation of the input.

* DPBloomFilter: A method that injects noise into the Bloom filter based on correlated sensitivity analysis, aiming to optimize the trade-off between utility and privacy under the differential privacy framework.

* Mangat\_filters: first generate a “polluted” set $\mathcal{S}'$ by randomly adding elements from the universe $\mathcal{U} \setminus \mathcal{S}$ to the original set $\mathcal{S}$ with probability $1 - p$ (i.e. $p = \frac{e^\epsilon}{e^\epsilon + 1}$).

* EBF-LDP: perturbs the query results using differential privacy methods before outputting them, ensuring differential privacy for each result.

* UltraFilter: first inserts all elements of the original set $S$ into a Bloom filter. Then, each bit of the Bloom filter is independently flipped to 1 with probability $\frac{e^\epsilon}{e^\epsilon + 1}$, while no bits are flipped to 0.

* DP-LBF：Applies the exponential mechanism to select the Bloom filter decision threshold while using an adaptive backup Bloom filter to preserve utility under differential privacy.
