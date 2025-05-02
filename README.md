# Traffic Light Detection Project

## Setup
1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [source_link] and place the files in the following structure:
   ```
   your_project/
   ├── dataset_train_rgb/
   │   └── train.yaml
   ├── dataset_test_rgb/
   │   └── test.yaml
   ```
4. Run the preprocessing scripts:
   ```bash
   python Create_Train_Files.py
   python Create_Test_Files.py
   ```
5. Train the model:
   ```bash
   python CNN_model_2.py
   ``` 