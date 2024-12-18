# IOB Slot Tagging

```plaintext
├── hw1_test.csv
└── hw1_train.csv
└── requirements.txt
├── run.py
├── BILSTM.py
└── submission.csv
```

Goal: Create a model that can tag movie-related utterances with IOB tags.

The BiLSTM.py file is the first model (BiLSTM) that was explored. The second and final BiGRU model is in the run.py file.

Addresses IOB slot tag classification using two sequential models: A bidirectional LSTM and its simpler counterpart, the bidirectional GRU model. The hyperparameters of the two models are tailored in such a way to get better validation loss and F1 scores.

To run:

```bash
pip install -r requirements.txt
python run.py hw2_train.csv hw2_test.csv submission.csv
```
