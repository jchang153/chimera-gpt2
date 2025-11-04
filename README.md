Fine-tuning GPT-2 to have separable parameters, i.e., a "Chimera" transformer where half of the parameters are trained on some objective $L_1$, and the other half on $L_2$. 

By ablating / patching one set of these parameters, we can (hopefully) eliminate the transformer's ability to perform well on that associated objective.
