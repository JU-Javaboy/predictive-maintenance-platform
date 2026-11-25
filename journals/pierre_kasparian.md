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