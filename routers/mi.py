# check pytorch version
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from models import *

from typing import Annotated, Optional
import datetime

from common.config import Tortoise


router = APIRouter(tags=["mi"])


def record_serializer(record):
    return {
        'measurement_id': record["measurement_id"],
        'measurement_date': str(record["measurement_date"]),
        'measurement_concept_id': record["measurement_concept_id"],
        'value_as_number': str(record["value_as_number"]),
        'concept_name': record["concept_name"],
        'concept_id': record["concept_id"],
        'person_id': record["person_id"],
    }

# dataset definition

class CSVDataset:
    # load the dataset
    def __init__(self, path):
        # store the inputs and outputs
        self.X = ...
        self.y = ...

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

@router.get("")
async def test():
    print(torch.__version__)
    conn = Tortoise.get_connection("default")
    
    patients = {}

    
    # weight stuff
    dead_people = await conn.execute_query('''
            SELECT
                m.measurement_id,
                m.measurement_date,
                m.measurement_concept_id,
                m.value_as_number,
                c.concept_name,
                c.concept_id,
                d.person_id,
                d.death_datetime,
                p.birth_datetime
            FROM
                omop.measurement m
            JOIN
                omop.death d ON m.person_id = d.person_id
            JOIN
                omop.concept c ON m.measurement_concept_id = c.concept_id
            JOIN
                omop.person p ON m.person_id = p.person_id
            where m.measurement_concept_id = 3025315
            --group by d.person_id, m.measurement_id, m.measurement_date, m.measurement_concept_id, m.value_as_number, c.concept_name, c.concept_id, 
            order by measurement_date asc
            
        ''')
    
    for row in dead_people[1]:
        if row["person_id"] not in patients:
            # 50 years = 18250 days
            # 60 years = 21900 days
            print("age at death: ", (row["death_datetime"] - row["birth_datetime"]).days)
            patients[row["person_id"]] = {
                "has_migraine": False,
                "has_diabetes": False,
                "has_asthma": False,
                'has_lung_cancer': False,
                'is_smoker': False,
                'dead': True,
                'has_cardiovascular_disease': False,
                'death_datetime': str(row["death_datetime"]),
                'birth_datetime': str(row["birth_datetime"]),
                'age_at_death': int((row["death_datetime"] - row["birth_datetime"]).days),
                'death_before_50': int((row["death_datetime"] - row["birth_datetime"]).days) < 18250,
            }
            patients[row["person_id"]]["weight_measurements"] = []
        patients[row["person_id"]]["weight_measurements"].append(record_serializer(row))
        
    living_over_50 = await conn.execute_query('''
        select
            p.person_id
        from 
            omop.person p 
        left join 
            omop.death d on p.person_id = d.person_id  
        where
            d.person_id is null and age(CURRENT_DATE, p.birth_datetime) > interval '50 years';
        ''')
    
    for row in living_over_50[1]:
        if row["person_id"] not in patients:
            patients[row["person_id"]] = {
                "has_migraine": False,
                "has_diabetes": False,
                "has_asthma": False,
                'has_lung_cancer': False,
                'is_smoker': False,
                'death_before_50': False,
                'dead': False,
                'has_cardiovascular_disease': False,
                'weight_measurements': []
            }
    
    diabetes_query = await conn.execute_query('''
            select
                co.condition_concept_id,
                co.person_id,
                c.concept_id,
                c.concept_name 
            from 
                omop.condition_occurrence co 
            join 
                omop.concept c on c.concept_id = co.condition_concept_id 
            where
                c.concept_id in (201826, 37200252, 37200254, 43530653, 44829882, 45533022, 45581355, 40386354);
        ''')
    
    # diabetes stuff
    # 201826 - diabetes
    # 37200252 - Type 2 diabetes mellitus with diabetic macular edema, resolved following treatment, left eye
    # 37200254 - Type 2 diabetes mellitus with diabetic macular edema, resolved following treatment, unspecified eye
    # 43530653 - Diabetic skin ulcer associated with type 2 diabetes mellitus
    # 44829882 - Diabetes with unspecified complication, type II or unspecified type, uncontrolled
    # 45533022 - Type 2 diabetes mellitus with diabetic peripheral angiopathy with gangrene
    # 45581355 - Type 2 diabetes mellitus with foot ulcer
    # 40386354 - Diabetes mellitus: [adult onset, with no mention of complication] or [maturity onset] or [non-insulin dependent]
    
    
    
    for row in diabetes_query[1]:
        if row["person_id"] in patients:
            patients[row["person_id"]]["has_diabetes"] = True
        else:
            continue
            patients[row["person_id"]] = {
                "has_migraine": False,
                "has_diabetes": True,
                "weight_measurements": []
            }
            
    # migraine stuff
    
    migraine_query = await conn.execute_query('''
            select
                co.condition_concept_id,
                co.person_id,
                c.concept_id,
                c.concept_name 
            from 
                omop.condition_occurrence co 
            join 
                omop.concept c on c.concept_id = co.condition_concept_id 
            where
                c.concept_id = 43530652;
        ''')
    
    
    for row in migraine_query[1]:
        if row["person_id"] in patients:
            patients[row["person_id"]]["has_migraine"] = True
        else:
            continue
            
            patients[row["person_id"]] = {
                "has_diabetes": False,
                "has_migraine": True,
                "weight_measurements": []
            }

    # 4051466 - childhood asthma
    asthma_query = await conn.execute_query('''
            select
                co.condition_concept_id,
                co.person_id,
                c.concept_id,
                c.concept_name 
            from 
                omop.condition_occurrence co 
            join 
                omop.concept c on c.concept_id = co.condition_concept_id 
            where
                lower(c.concept_name) like '%asthma%';
                ''')
    
    for row in asthma_query[1]:
        if row["person_id"] in patients:
            patients[row["person_id"]]["has_asthma"] = True
        else:
            continue
            
            patients[row["person_id"]] = {
                "has_diabetes": False,
                "has_migraine": True,
                "weight_measurements": []
            }
                
    
    # 4115276 - lung cancer
    lung_cancer_query = await conn.execute_query('''
            select
                co.condition_concept_id,
                co.person_id,
                c.concept_id,
                c.concept_name 
            from 
                omop.condition_occurrence co 
            join 
                omop.concept c on c.concept_id = co.condition_concept_id 
            where
                lower(c.concept_name) like '%lung cancer%';
                ''')
    
    for row in lung_cancer_query[1]:
        if row["person_id"] in patients:
            patients[row["person_id"]]["has_lung_cancer"] = True
        else:
            continue
            
            patients[row["person_id"]] = {
                "has_diabetes": False,
                "has_migraine": True,
                "weight_measurements": []
            }
            
    
    # is smoker
    smoker_query = await conn.execute_query('''
            select
                o.observation_id,
                o.observation_concept_id,
                c.concept_id,
                c.concept_name,
                o.person_id
            from 
                omop.observation o
            join 
                omop.concept c on c.concept_id = o.observation_concept_id
            where
                lower(c.concept_name) like '%smoke%';
                ''')
    
    for row in smoker_query[1]:
        if row["person_id"] in patients:
            patients[row["person_id"]]["is_smoker"] = True
        else:
            continue
            
            patients[row["person_id"]] = {
                "has_diabetes": False,
                "has_migraine": True,
                "weight_measurements": []
            }
             
    
    # cardiovascular disease
    # 36304558 - Cardiovascular disease
    
    cardio_query = await conn.execute_query('''
            select
                o.observation_id,
                o.observation_concept_id,
                c.concept_id,
                c.concept_name,
                o.person_id
            from 
                omop.observation o
            join 
                omop.concept c on c.concept_id = o.observation_concept_id
            where
                lower(c.concept_name) like '%cardiovascular%';
        ''')
    
    for row in cardio_query[1]:
        if row["person_id"] in patients:
            patients[row["person_id"]]["has_cardiovascular_disease"] = True
        else:
            continue
            
            patients[row["person_id"]] = {
                "has_diabetes": False,
                "has_migraine": True,
                "weight_measurements": []
            }
    
    # export to data.json

    with open('data.json', 'w') as outfile:
        json.dump(patients, outfile, indent=4)
    
    return "cool"

    # cancer stuff
    # 4194405 - cancer confirmed
    # 443392 - Malignant neoplastic disease
    # 40572468 - cancer confirmed (again)
    
    """cancer_query = await conn.execute_query('''
            select
                co.condition_concept_id,
                co.person_id,
                c.concept_id,
                c.concept_name 
            from 
                omop.condition_occurrence co 
            join 
                omop.concept c on c.concept_id = co.condition_concept_id 
            where
                c.concept_id in (4194405, 443392, 40572468);
        ''')"""

@router.get("/to_csv")
async def to_csv():
    conn = Tortoise.get_connection("default")
    
    # get patients data from data.json
    with open('data.json') as json_file:
        patients = json.load(json_file)
        
    # convert to csv
    with open('data.csv', 'w') as outfile:
        outfile.write("person_id,has_migraine,has_diabetes,has_asthma,is_smoker,has_cardiovascular_disease,death_before_50,has_lung_cancer\n")
        for key in patients:
            outfile.write(str(key) + "," + str(patients[key]["has_migraine"]) + "," + str(patients[key]["has_diabetes"]) + "," + str(patients[key]["has_asthma"]) + "," + str(patients[key]["is_smoker"]) + "," + str(patients[key]["has_cardiovascular_disease"]) + "," + str(patients[key]["death_before_50"]) + "," + str(patients[key]["has_lung_cancer"]) + "\n")
            

# Define the model
class DualOutputModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DualOutputModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

@router.get("/train")
async def train():
    df = pd.read_csv("data.csv")
    
    # drop person_id
    #df = df.drop(columns=["person_id"])
    
    # Map 'True' and 'False' to 1 and 0 in the 'death_before_50' column
    df['death_before_50'] = df['death_before_50'].map({True: 1, False: 0})
    
    df['has_migraine'] = df['has_migraine'].map({True: 1, False: 0})
    df['has_diabetes'] = df['has_diabetes'].map({True: 1, False: 0})
    df['has_asthma'] = df['has_asthma'].map({True: 1, False: 0})
    df['has_lung_cancer'] = df['has_lung_cancer'].map({True: 1, False: 0})
    df['is_smoker'] = df['is_smoker'].map({True: 1, False: 0})
    df['has_cardiovascular_disease'] = df['has_cardiovascular_disease'].map({True: 1, False: 0})
    
    # Define input size, hidden size, and output size based on your data
    input_size = 5
    hidden_size = 15
    output_size = 1  # Binary classification (died before 50 or not)
    
    # Split the data into training and testing sets
    X = df[['has_migraine', 'has_diabetes', 'has_asthma', 'is_smoker', 'has_cardiovascular_disease']].values
    y = df[['death_before_50']].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)

    # Ensure y_train has two columns (matches output_size)
    if y_train.shape[1] != output_size:
        raise ValueError("Number of columns in y_train does not match the output_size of the model.")


    model = DualOutputModel(input_size, hidden_size, output_size)  
      
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Assuming X_train and y_train are your input features and labels
    # Convert them to PyTorch tensors

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)

    # Train the model
    epochs = 120
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Print some information during training
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
        
        
    torch.save(model.state_dict(), 'model.pth')
    
    # Testing the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.Tensor(X_test))
        predicted_labels = (test_outputs >= 0.5).float().view(-1).numpy()

    print(test_outputs)
    # show predicted labels as 2d array
    #print(predicted_labels)
    
    # convert predicted labels to 2d array from 1d array
    #predicted_labels = predicted_labels.reshape(-1, 2)
    

    
    accuracy = accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels, average='macro')
    recall = recall_score(y_test, predicted_labels, average='macro')
    
    
    fig, ax = plt.subplots()
    
    # convert y_test to 1d array
    y_test = y_test.reshape(-1)

    print(y_test)
    print(predicted_labels)
    
    print(X_test, "x_test")
    
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    
    # find out what percentage of 0s are correct
    # under .2 is kinda sure
    
    zero_winrate = 0
    for i in range(len(predicted_labels)):
        if y_test[i] == 0:
            if test_outputs[i] < 0.2:
                zero_winrate += 1
        else:
            if test_outputs[i] < .2:
                zero_winrate -= 1    
            
                
                
    zero_winrate = zero_winrate / len(predicted_labels)
    print("zero winrate: ", zero_winrate)

    # get only numbers from test_outputs, 2 decimal, normal array
    test_outputs = test_outputs.numpy().round(2)

    # convert test_outputs to 1d array
    test_outputs = test_outputs.reshape(-1)

    # Creating a table
    table_data = [X_test,predicted_labels, y_test, test_outputs]
    table = ax.table(cellText=table_data, loc='center')

    # Styling the table
    table.set_fontsize(13)
    table.scale(1.25, 1.25) 
    
    
    # dont show axis
    ax.axis('off')
    
    plt.show()
    
@router.get("/predict")
async def predict(has_migraine: bool = False, has_diabetes: bool = False, has_asthma: bool = False, is_smoker: bool = False, has_cardiovascular_disease: bool = False):
    model = DualOutputModel(5, 15, 1)
    model.load_state_dict(torch.load('model.pth'))
    
    # convert to tensor and change to float
    has_migraine = float(has_migraine)
    has_diabetes = float(has_diabetes)
    has_asthma = float(has_asthma)
    is_smoker = float(is_smoker)
    has_cardiovascular_disease = float(has_cardiovascular_disease)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.Tensor([has_migraine, has_diabetes, has_asthma, is_smoker, has_cardiovascular_disease]).float())
        print(test_outputs)
    return test_outputs.item()