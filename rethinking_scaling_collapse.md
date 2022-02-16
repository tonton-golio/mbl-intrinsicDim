# rethinking scaling collapse


for an evolutionary approach, we need to define a cost function.

The cost function should be a distance metric:
	* if L1 we consider absolute distance, meaning distances [1,3] results in the same cost as [2,2]
	* if L2, we significantly punish outliers, and therefore prefer to have all "relatively" close.

This cost function might weight data-points dependent on distance from the persumed criticality. 
	* if we include such a weighting, we might be able to use the entire data-set, rather than just taking into account points around the criticality.




































