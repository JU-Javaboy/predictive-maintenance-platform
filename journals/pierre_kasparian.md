# Pierre Kasparian — Project Journal

16 nov week :
- data exploration
- business understanding

23 nov week : 
- data / feature exploration
- choose the best model for the problem (random forest fits well for classification)
- wrong way to do with HDBSCAN (noticed afterwards dataset labelisation concerned our problem and not C2's one)
- differents trials with different features, finally ended up with z score, recently seen and resolution_gap features, achieving 100% accuracy, which is suspicious. Keeping it in mind for later
- implemented functions for data transformations at inference time in the project, not only in notebooks

27 nov
- looked into some guardrails for the model in case of low confidence. Conclude that we can make quite accurate prediction based on fixed rules (ex : z_score_temperature >80 , resolution drop giving nb of decimals below 2. 
- Looking at the visulisation, it looks like the sensor spike are 100% coming from the temperature. That can probably cause the model to overfit. We will have to set threshold to other sensors, by looking at the visualisation in explo_models_guardrails.ipynb