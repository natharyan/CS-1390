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
        - Within type, there are only two categories that have fraudulent cases.
        - For fraud transactions, the new - old account balance for the sender is very low or negative at a high frequency, the same for the receiver is a low positive value at a high frequency.
        - In the same for non fraud transactions, for the sender its centred close to 0 at a high frequency, for the receiver it is mostly a low positive value at a high frequency.
        - (About bank account categories, etc.)
    - Hyperparameter tuning, model variations:
        - DT pruning.
        - 