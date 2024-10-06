Report
● Format:
○ Submit a written report in PDF format named report.pdf.
● Content:
○ Introduction: Briefly describe the problem and your approach.
○ Methodology: Explain the algorithms and techniques you used.
○ Experimentation:
■ Discuss the different experimentations you conducted.
■ Explain any hyperparameter tuning or model variations.
○ Results:
■ Present the evaluation metrics and discuss the performance of your models.
■ Include any relevant tables, graphs, or charts.
○ Challenges:
■ Discuss any difficulties faced and how you overcame them. ○ Conclusion:
■ Summarize your findings and suggest possible improvements. ○ References: Include any references used.

Decision Trees

# Introduction
- Problem: Online payment fraud detection using the account id of the transaction sender and receiver, the amount of transaction, the change in bank account balance and the type of transaction.
- Algorithms and techniques used:
    - I have implemented the decision tree algorithm for this problem. 
    - Techniques used - 
- Experimentation:
    - On exploring the dataset I found out the the following:
        - Engine size and fuel consumption share a linear relationship.
        - CO-Emissions and fuel consumption share a linear relationship.
        - Cylinders and fuel consumption dont have a meaningful relationship, there are multiple fuel consumption values for the same cylinder value.
    - Hyperparameter tuning, model variations:
        - Model 1 - with single engine emissions feature
        - Model 2 - with both engine emissions and coemissions features
        - I used standardization instead of normalisation and got an immense improvement in the performance. Got 0.5440923114036047 on the training dataset and 0.30161212190512865 on the validation dataset.
