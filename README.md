# Response_inhibition_models
# An activation threshold model for response inhibition  

An activation threshold model (ATM) and horse-race model (HRM) presenting conceptual frameworks to explain behavioural and neurophysiological data from response inhibition experiments  requiring partial movement cancellation. The models comprise both facilitatory and inhibitory processes that compete upstream of motor output regions. Summary statistics (means and standard  deviations) of predicted muscular and neurophsyiological data are fitted to equivalent experimental measures by minimizing a Pearson Chi-square statistic.  

Prerequisites: scipy, numpy, sys  

To run ATM on Go and Partial Stop trials using a single facilitation curve: `python ARI_task ATM_first` 
To run ATM on Partial Stop trials with an additional facilitation curve using manual input for optimized parameters from single facilitation curve: `python ARI_task ATM_second` 
To run HRM on Go and Partial trials: 'python ARI_task HRM'  

Accompanying manuscript at XX
