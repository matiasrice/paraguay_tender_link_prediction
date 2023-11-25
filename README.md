# Link Prediction on Paraguay’s Public Tender Participants

## Project Overview
In many countries, the government stands as one of the largest entities engaging in the procurement of services and the purchase of products. The initiation of this procurement phase typically involves a tendering process. Unfortunately, there are instances where the number of companies participating in these processes is limited. This challenge could be addressed by implementing a strategy wherein companies already providing services to other government institutions are made aware of additional tender opportunities. 

This approach can be conceptualized as a link prediction problem, where the goal is to forecast future or plausible links between a supplier and an institution. To operationalize this concept, we extract features from the public tender information in Paraguay and represent the supplier-institution relationship using a bipartite graph structure. We harness the capabilities of two contemporary models, GraphSAGE and GAT, to enhance the classification task. The experimental results demonstrate a high level of precision following the application of these methods.

## Directory Structure
You can find the data wrangling and preprocess notebook in the [notebooks folder](./notebooks/). The final modeling notebook is located [data_modeling.ipynb](data_modeling.ipynb) file.

If you want to run the whole pipeline, from data extraction to data modeling, you should go in the following order:

1. [data_collection.ipynb](./notebooks/Data_collection.ipynb)
2. [collected_data_wrangling.ipynb](./notebooks/Collected_data_wrangling.ipynb)
3. [data_modeling.ipynb](./data_modeling.ipynb)

## Installation
Our project requires specific packages to be installed. The `requirements.txt` file lists the Python dependencies required for this project. If you set up a new environment or if you want to ensure compatibility, you can use this file to install the correct versions of the necessary packages.

Here is an option to start:

1. **Clone the repository**:
   
```bash
$git clone https://github.com/matiasrice/paraguay_tender_link_prediction.git
$cd paraguay_tender_link_prediction
```

2. **Create a virtual environment**:

For Unix and MacOS:

```bash
$python3 -m venv myenv
$source myenv/bin/activate
```

For Windows:

```bash
$python -m venv myenv
$myenv\Scripts\activate
```
Note: Deactivate the virtual environment when done:
After you're done working, remember to deactivate the virtual environment to return to your global Python environment.

```bash
$deactivate
```

3. **Install the required packages**:
```bash
$pip install -r requirements.txt
```
Your environment is now ready.

## Disclaimer
This work was developed as a class project. The main goal of the project was not to structure the repo using all the best practices, but to get some results and generate a report, the repo is just aimed to make the code available. We realize that there is a lot room for improvement in the structure of the repo. However, because of the time-constraint, we will not focus on that right now. 

## More details
You can find the final summarized report [here](./report/).
