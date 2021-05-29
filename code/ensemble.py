import numpy as np
import pandas as pd

model_A = np.load('/opt/ml/code/prediction/npy_output/electra-base-v3-2_kfold.npy')
model_B = np.load('/opt/ml/code/prediction/npy_output/xlm-roberta-large_kfold.npy')
model_C = np.load('/opt/ml/code/prediction/npy_output/BEST.npy')

result = np.stack([model_A, model_B, model_C], axis=0)
result = result.mean(axis=0)
result = result.argmax(axis=-1)

pred_answer = np.array(result).flatten()

output = pd.DataFrame(pred_answer, columns=['pred'])  
output.to_csv(f'./prediction/ensemble7.csv', index=False)
