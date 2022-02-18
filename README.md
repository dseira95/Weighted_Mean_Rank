# Weighted Mean Rank (Incident/Dynamic AUC)
The prognostic models are well known to be used for investigating patient outcomes in relation to their disease characteristics. If these models are working correctly, hence, making accurate predictions, then we are able to identify the patients who are at greater risk of a specific event, such as death. With this information it is possible to classify the groups of patients as high and low risk and therefore, help the ones who need it the most. Given that the performance of these classifications are not stable and change over time, we need statistical methodologies that characterize the time-varying accuracy of prognostic models when used for dynamic decision making.

As stated by Bansal and Heagerty, there are several existing statistical methods for evaluating prognostic models with simple binary outcomes, but methods appropriate for survival outcomes are less well known and require time-dependent extensions of sensitivity and specificity to fully characterize longitudinal biomarkers or models.

Therefore, the weighted mean rank methodology is particularly important as it allows for appropriate handling of censored outcomes, which are commonly seen in event time data. In this case, the focus of the function is to calculate the Incident/Dynamic Area Under the Curve (AUC), which can help determine the point in time where the performance of the patients classification has lowered; suggesting it would be prudent to update the data of the patients.

## Example
The data used for the example is the Primary Biliary Cirrhosis Mayo clinic dataset included in the survival package in R. For more information about the data visit https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pbc.html. 

In the example, the WMR is used to compute the Incident/Dynamic AUC and compare two models, one with 4 covariates (age, edema, albumin, and log_protime) and the other with 5 covariates (4 previous + log_bili). And from the resulting graph of the AUC_I/D(t) v Time and with the NNE smoother included it can be seen that the serum bilirubin is an important factor to consider for accurate predictions, since its inclusion in the model lead to a greater classification performance. Furthermore, a considerable decay in AUC is portrayed after some time, emphasizing the importance of updating patient data to stabilize the classification performance of the model.

## References
Heagerty, P. J., &amp; Zheng, Y. (2005, February 28). Survival model predictive accuracy and ROC curves. 
Wiley Online Library. Retrieved February 16, 2022, from https://onlinelibrary.wiley.com/doi/10.1111/j.0006-341X.2005.030814.x 

Saha-Chaudhuri, P., &amp; Heagerty, P. J. (2012, June 25). Non-parametric estimation of a time-dependent predictive accuracy curve. 
OUP Academic. Retrieved February 16, 2022, from https://academic.oup.com/biostatistics/article/14/1/42/250188 

Bansal, A., &amp; Heagerty, P. J. (2018). A tutorial on evaluating the time-varying discrimination accuracy of survival models used in dynamic decision making. 
Medical Decision Making, 38(8), 904–916. https://doi.org/10.1177/0272989x18801312 
