# Optimizing-ECommerce-Pricing-Model
This repository encapsulates G:64's complete work for the course's project Milestone-2 for CSF305: PoPL 
Contributors are:
  1. Kuhu Gupta (2020B4A71524G)
  2. Prathamesh Tiwari (2021A7PS2834G)
  3. Sejal Sarada (2020B4A71849G)
  4. Tejas Saraogi (2021A7PS2835G)

# Problem Statement: Optimizing E-Commerce Pricing Strategies Using Probabilistic Programming
## Introduction
In the dynamic landscape of e-commerce, pricing strategies play a pivotal role in determining a business’s profitability. The challenge lies in striking a delicate balance: setting prices that attract customers, maximize revenue, and adapt to ever-changing market conditions. Traditional deterministic pricing models fall short in capturing the inherent uncertainty and variability associated with customer behavior, sales patterns, and external factors. Our goal is to develop a sophisticated probabilistic pricing model that optimizes revenue while considering uncertainty and variability. This model will be implemented using both **Pyro** (a probabilistic programming language) and **Python** (a general-purpose language). Our focus extends beyond mere comparison; we aim to dissect the technical intricacies and POPL aspects inherent in each approach.
## Problem Context
* **E-Commerce Domain**: Our focus is on e-commerce platforms where a vast array of products are sold online. In this domain, product prices directly impact sales and profitability. Suboptimal pricing can lead to missed revenue opportunities or erode profit margins.
* **Uncertainty and Variability**: Traditional deterministic models fall short in capturing the dynamic nature of customer behavior, market conditions, and sales data.
* **Probabilistic Approach**: Our solution leverages probabilistic programming to model uncertainty explicitly.
* **Multifaceted Factors**: Pricing decisions must account for:
  * Customer Behavior: How do customers respond to price changes? What are their preferences and sensitivities?
  * Sales Data: Historical sales patterns, seasonal trends, and product life cycles.
  * Market Conditions: Competitor pricing, demand fluctuations, and economic shifts.
## Goals and Technical Goals
The primary objective is to optimize pricing strategies by leveraging probabilistic programming. We seek to find price points that:
* **Maximize Profit**: Balancing revenue and costs.
* **Adapt Dynamically**: Responding to changing market dynamics.
* **Account for Uncertainty**: Acknowledging that our knowledge of customer behavior and external factors is inherently uncertain.
* **Probabilistic Modeling**: Develop a pricing model that:
* **Expressiveness**: Represents complex relationships probabilistically.
* **Incorporates Uncertainty**: Models customer preferences, demand fluctuations, and external factors stochastically.
POPL Integration: Embeds POPL concepts within the probabilistic framework.
* **Efficiency and Scalability**:
  Pyro Efficiency: Evaluate Pyro’s computational efficiency for large-scale pricing optimization.
  Python Scalability: Assess Python’s ability to handle extensive datasets and complex models.
* **Abstraction and Reusability**:
  Pyro Abstraction: Investigate how Pyro abstracts low-level details, allowing focus on high-level pricing strategies.
* **Code Reusability**: Identify reusable components across both Pyro and Python implementations.

## Approach
We propose a probabilistic pricing model that:
* **Expressiveness**: Captures complex relationships between pricing variables using probabilistic constructs.
* **Efficiency**: Balances computational efficiency with model accuracy.
* **Abstraction**: Abstracts away low-level details, allowing us to focus on high-level pricing strategies.
* **Community and Ecosystem**: Evaluates the support, documentation, and community engagement for both Pyro and Python.

## Comparison: Pyro vs. Python
* **Pyro (Probabilistic Programming)**:
  Expressiveness: How well does Pyro allow us to model uncertainty and dependencies?
  Efficiency: Is Pyro computationally efficient for our use case?
  Abstraction: Does Pyro simplify complex probabilistic models?
  Community and Ecosystem: What resources and libraries are available for probabilistic programming in Pyro?
* **Python (Traditional Approach)**:
  Expressiveness: How flexible is Python for expressing pricing models?
  Efficiency: Can Python handle large-scale pricing optimization?
  Abstraction: How much manual effort is required for model development?
  Community and Ecosystem: What Python libraries support pricing analytics?
  
## Conclusion
By comparing Pyro and Python, we aim to provide insights into the best tool for building robust probabilistic pricing models. Our exploration will shed light on the trade-offs between expressiveness, efficiency, and community support. Ultimately, we strive to empower e-commerce businesses with data-driven pricing decisions that enhance profitability.

# Software Architecture of our Solution for Optimizing Pricing Strategies using Pyro vs. Python : 

![SoftwareArchitectureFlowchart](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/4c2e5d0d-f902-411a-a698-e18297fc8acc)

The software architecture for our pricing optimization solution involves a combination of components and their interactions.

* Components:
  * Data Ingestion: Collects data from various sources (e.g., sales data, competitor prices, customer segments).
  * Feature Engineering: Extracts relevant features (e.g., seasonality, promotions) from raw data.
  * Pricing Models:
    * Pyro Implementation: Utilizes probabilistic programming to model uncertainty and dependencies.
    * Traditional Python Implementation: Uses linear regression or other traditional techniques.
  * Inference Engines:
    * Pyro: Performs stochastic variational inference (SVI) to estimate model parameters.
    * Traditional Python: Calculates coefficients using least squares.
  * Evaluation Metrics: Measures accuracy (e.g., mean squared error) and execution time.
  * Decision Engine: Determines optimal pricing strategies based on model outputs.
  * Feedback Loop: Incorporates real-world feedback to continuously improve the models.

* Interactions:
  Data flows from ingestion to feature engineering.
  Feature-engineered data feeds into both pricing models.
  Inference engines estimate model parameters.
  Evaluation metrics assess model performance.
  Decision engine selects pricing strategies.
  Feedback loop updates models based on actual sales data.

* Architecture Types:
  * Client-Server: Interaction between components (e.g., pricing models, decision engine).
  * Pipeline: Sequential flow of data and processing steps.
  * Microservices: Decoupled components for scalability and maintainability.

* Trade-offs:
  * Pyro: Offers expressiveness and uncertainty modeling but may be computationally intensive.
  * Traditional Python: Simpler but less flexible for complex relationships.

In summary, our solution combines probabilistic programming (Pyro) with traditional techniques to optimize pricing strategies. The architecture balances accuracy, efficiency, and interpretability.










# PoPL Aspects of our Solution to Optimizing Pricing Model : 

1) Pyro-based Model

![snippet1](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/f6023f60-792d-4aa2-9f1e-809bc19e615f)


Modularity: The code starts by importing external libraries, demonstrating modularity by using pre-built modules for data manipulation (pandas), numerical operations (torch), probabilistic programming (pyro), and machine learning utilities (MinMaxScaler, train_test_split, mean_squared_error).

![snippet2](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/1f6c6721-386b-4571-aa17-1067108c7d91)


Abstraction: The code abstracts away the details of loading the dataset into a Pandas DataFrame. The use of pd.read_csv abstracts the complexity of reading data from a CSV file.

![snippet3](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/6f15bcfe-ecd3-4079-8381-686497dd13d1)


Abstraction: The use of MinMaxScaler abstracts the scaling process for numeric features. This encapsulates the details of normalization. 
Modularity: The preprocessing steps are organized into a separate section, promoting modularity by isolating data preparation logic.

![snippet4](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/ee57367e-2abe-4c4c-a7f8-f8c437c69483)


Abstraction: The code abstracts the process of splitting the data into training and testing sets using train_test_split. This hides the details of data partitioning.

![snippet6](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/f65164d9-397e-4e7d-ae09-e5b6fce44098)


Abstraction: The PyroPricingModel class abstracts the definition of the probabilistic pricing model. This encapsulates the details of the model architecture. 
Modularity: The model definition is modular, with a clear separation between model architecture and training logic.

![snippet7](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/f628b5a9-b4c7-4fb6-ab32-dd0fde7a6806)


Modularity: The training logic is encapsulated within the train_model method, promoting modularity by separating training details from the model definition. 
Abstraction: The training process is abstracted away into a function, hiding the details of how the model is trained.

![snippet8](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/76a6437f-aa2d-49ba-8533-5a55a7793b1a)


Modularity: The prediction and evaluation steps are encapsulated in the predict_prices method, promoting modularity by isolating prediction and evaluation logic. 
Abstraction: The details of obtaining predictions and evaluating model performance are abstracted into functions, making the main script more readable.

The code demonstrates principles of programming languages (POPL) aspects such as modularity, abstraction, and organization of logic. The use of classes, methods, and library functions helps make the code more structured and readable.

2) Traditional Python Code

![snippet9](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/fb869f7f-61a0-4add-9b76-c6a35f737f1c)

Modularity: The code starts by importing external libraries, demonstrating modularity by using pre-built modules for data manipulation (pandas), machine learning model (LinearRegression), and model evaluation (train_test_split, mean_squared_error).

![snippet10](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/77a97c38-05c7-4da8-9d19-d09682eddf46)



Abstraction: The code abstracts away the details of loading the dataset into a Pandas DataFrame using pd.read_csv. This encapsulates the complexity of reading data from a CSV file.

![snippet11](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/a3d348a7-dc5d-4ca7-8fc4-c1b74e9b0051)


Abstraction: The use of pd.get_dummies abstracts the process of one-hot encoding categorical features. This hides the details of the encoding process.
Modularity: The preprocessing steps are organized into a separate section, promoting modularity by isolating data preparation logic.


![sinppet12](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/0f22e927-881f-4e50-9ff8-4a85d069dd4e)


Abstraction: The code abstracts the process of splitting the data into training and testing sets using train_test_split. This hides the details of data partitioning.


![snippet13](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/0a41a2d5-e912-4674-8ff3-7d0465a7b7ec)

Abstraction: The code abstracts the training of a linear regression model using LinearRegression and fit method. This encapsulates the details of the training process.

![snippet14](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/cec6ab83-51e6-4875-aa4c-30ae09e01678)



Abstraction: The code abstracts the prediction process using the trained model's predict method. This hides the details of the prediction process.

![snippet15](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/15d0d410-47d7-4e8d-95de-313cdc374b5e)


Modularity: The evaluation logic is encapsulated in the calculation of mean squared error, promoting modularity by isolating evaluation details.
Abstraction: The details of calculating and printing mean squared error are abstracted into a few lines, making the code more readable.
![snippet16](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/97660514-3010-44fc-9dea-df1ff6b06374)


Modularity: The code encapsulates the process of extracting and printing pricing coefficients, promoting modularity by isolating this logic. 
Abstraction: The details of obtaining and printing pricing coefficients are abstracted into a few lines, making the code more readable.









# Comparison between Pyro and Python : 

In this project, we undertook the task of optimizing an E-Commerce pricing strategy using probabilistic programming, implementing the solution in both Python with a standard probabilistic programming library and Pyro, a probabilistic programming language built on PyTorch. The comparison aimed to evaluate various aspects, including language paradigms, syntax, inference methods, ease of use, expressiveness, integration with deep learning, community support, and alignment with principles of programming concepts.
The runtime performance of the two models (Pyro probabilistic model vs. Traditional Linear Regression model) depended on various factors, including the size of our dataset, the complexity of the models, and the efficiency of the underlying libraries.

Following are considerations for each approach :

* Pyro Probabilistic Model:
  Pros:
  Probabilistic programming allows modeling of uncertainties, which might be valuable for certain types of problems.
  Pyro and PyTorch are highly optimized for GPU acceleration, making it suitable for large-scale data and complex models.
  Cons:
  The probabilistic model may have a longer training time due to the complexity of the inference process.
  The training process involves sampling, which can be computationally expensive.
* Traditional Linear Regression Model:
  Pros:
  Linear regression is a simpler model and can be computationally efficient, especially for small to medium-sized datasets.
  The training process is typically faster compared to probabilistic models.
  Cons:
  Assumes a linear relationship between features and target, which might not capture complex patterns.
  Doesn't model uncertainties explicitly.

## Considerations:
  * Dataset Size:
    For small to medium-sized datasets, the difference in runtime might not be significant.
    For large datasets, especially if you have access to GPU acceleration, the Pyro model might be more scalable.
  
  * Model Complexity:
    If the relationship between features and target is linear and the dataset is not too complex, a linear regression model might perform well.
    If the relationship is nonlinear or involves uncertainties, the probabilistic model may provide better results.
  
  * Computational Resources:
    The probabilistic model might benefit from GPU acceleration. If you have access to GPUs, it could potentially outperform the traditional linear regression model.
In summary, the runtime performance depends on your specific use case and the characteristics of your data. It's recommended to test both models on your specific dataset to determine which one performs better in terms of both accuracy and runtime. For small to medium-sized datasets, the difference in runtime might not be a critical factor, and you can choose the model based on its ability to capture the underlying patterns in the data.


![Comparison](https://github.com/SejalSarada/Optimizing-E-Commerce-Pricing-Model/assets/77984669/dce12346-8884-4d8d-b1bc-9bf6b47d2fdf)



## Findings:

1. Language Paradigm and Syntax:
Python:
Utilizes the syntax of the chosen probabilistic programming library.
Pyro:
Has its own syntax based on PyTorch, designed for expressiveness and flexibility.

2. Inference Methods:
Python:
Depending on the library chosen, common methods include MCMC and VI.
Pyro:
Supports various inference algorithms, including MCMC, VI, and Sequential Monte Carlo.

3. Ease of Use:
Python:
Generally easy for users familiar with Python.
Pyro:
Learning curve may be steeper due to PyTorch integration, but provides more control and flexibility.

4. Expressiveness:
Python:
May have limitations in expressing complex probabilistic models.
Pyro:
Offers high expressiveness due to its integration with PyTorch, allowing for more intricate models.

5. Integration with Deep Learning:
Python:
May require additional libraries for seamless integration with deep learning.
Pyro:
Tightly integrated with PyTorch, providing native support for deep probabilistic models.

6. Community Support:
Python:
Wide community support for general probabilistic programming libraries.
Pyro:
Growing community with a focus on deep probabilistic programming in PyTorch.









# Verification and Results :

Comparison tests were conducted between the pyro-based model and traditional python model. ‘Comparion.py’ was used in order to make statistical comparisons between the results provided by the 2 models.

**We have added our results as graphs in the /doc folder, be welcome to check them out!**

## Conclusion:
  Based on the comparison, both Python with a standard probabilistic programming library and Pyro offer unique advantages and considerations for implementing probabilistic programming in the context of E-Commerce pricing strategies. Python provides a more straightforward entry point for users familiar with the language, while Pyro, with its integration with PyTorch, offers enhanced expressiveness, particularly for deep probabilistic models.
  The choice between the two should be driven by project requirements, the complexity of the probabilistic model, and the familiarity of the team with deep learning concepts. Python may be preferable for simpler models and a quicker implementation, while Pyro becomes a compelling choice for projects requiring advanced expressiveness and integration with deep learning frameworks.
  In conclusion, the selection between Python and Pyro should align with the specific needs and goals of the E-Commerce pricing strategy project, emphasizing the principles of abstraction, modularity, and readability in probabilistic programming design.










# Potential for Future Work :

Given more time, we would explore:
Inference Algorithms: Investigate advanced probabilistic inference techniques.
Domain-Specific Extensions: Extend the model for specific e-commerce niches (e.g., fashion, electronics).
Human-Centric Pricing: Incorporate user preferences and behavioral data.
