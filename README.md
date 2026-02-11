# \# EECS590 Capstone — Reinforcement Learning for Hospital Readmission Planning



### \## Project Overview



This capstone project investigates how Reinforcement Learning can be used to model and optimize sequential clinical decision-making processes that influence hospital readmissions.



Hospital readmissions remain one of the most persistent and costly problems in healthcare systems. Patients discharged after treatment frequently return within 30 days due to complications, inadequate follow-up care, medication mismanagement, or premature discharge decisions. These readmissions negatively impact patient outcomes and create significant financial burdens for healthcare providers.



The goal of this project is to design reinforcement learning agents capable of learning optimal intervention strategies that minimize avoidable hospital readmissions while maximizing long-term patient recovery outcomes.



Version 1 establishes the computational and algorithmic foundation required to pursue this objective throughout the semester.



---



### \## Capstone Problem Statement



Hospital discharge planning is not a single decision but a sequence of interdependent clinical choices. Providers must determine discharge timing, medication plans, rehabilitation referrals, patient education strategies, and follow-up care intensity. Each of these decisions affects a patient’s recovery trajectory and their probability of readmission.



Traditional statistical models identify associations between risk factors and readmissions but do not optimize sequential care strategies. Reinforcement Learning provides a framework for modeling healthcare as a decision process in which an agent learns from outcomes and improves intervention policies over time.



This project formulates hospital readmission pathways as a Markov Decision Process (MDP), where patient recovery states evolve stochastically based on clinical actions and intervention strategies.



---



### \## MDP Formulation of the Readmission Problem



States represent abstracted patient recovery stages following discharge, ranging from stable recovery to high-risk deterioration states.



Actions represent intervention strategies such as conservative monitoring, intensive follow-up care, rehabilitation escalation, or early clinical intervention.



Transition probabilities model uncertainty in patient response to treatment and post-discharge care.



Rewards are structured to penalize inefficient care pathways while strongly incentivizing successful recovery without readmission.



Terminal states represent either successful long-term recovery or hospital readmission events.



---



#### \## Version 1 Implementations



Version 1 focuses on building the foundational reinforcement learning framework required for later algorithmic expansion.



##### \### Environment



A tabular foundation environment was implemented to simulate patient recovery dynamics. The environment includes stochastic transitions, intervention actions, and outcome-based rewards.



##### \### Dynamic Programming



Policy Iteration was implemented to compute optimal policies within the foundation environment. This includes iterative policy evaluation and greedy policy improvement.



##### \### Training Framework



A command-line training interface was developed to allow reproducible policy learning under configurable hyperparameters.



##### \### Evaluation Framework



A simulation pipeline evaluates learned policies across multiple stochastic episodes, reporting cumulative returns and terminal recovery rates.



##### \### Visualization



Value functions and learned policies are visualized to interpret agent decision behavior across recovery states.



---



##### \## Repository Structure



src/ contains all reinforcement learning source code, including environments, agents, MDP definitions, and CLI pipelines.



outputs/ stores trained policies, value functions, evaluation metrics, and visualization plots.



scripts/ includes utilities such as policy visualization.



tests/ is reserved for future environment and agent validation tests.



requirements.txt defines project dependencies.



---



##### \## How to Run the Project



Activate the virtual environment and set the Python path:



```powershell

.\\.venv\\Scripts\\Activate.ps1

$env:PYTHONPATH="src"

```



Train the agent:



```powershell

python -m eecs590\_capstone.cli.train

```



Evaluate the learned policy:



```powershell

python -m eecs590\_capstone.cli.eval

```



Run multiple evaluation simulations:



```powershell

1..5 | ForEach-Object {

&nbsp; python -m eecs590\_capstone.cli.eval --episodes 2000 --seed $\_

}

```



---



##### \## Version 1 Outputs



Tracked outputs include:



Policy kernel

Value function estimates

Training metadata

Evaluation metrics

Policy visualization plots



These artifacts demonstrate successful policy convergence and environment simulation stability.



---



##### \## Future Work



Future versions of this capstone will extend the framework to include:



Value Iteration

Monte Carlo learning

Temporal Difference learning

SARSA and Q-Learning

Eligibility Traces

Exploration strategies

Function approximation methods



The environment will also be expanded to incorporate higher-dimensional patient features and real healthcare datasets.



---



###### \## Author



Christianah Jemiyo

PhD Student, Artificial Intelligence

University of North Dakota



---



