import os
import numpy as np
import torch
from lstm import CustomLSTM
from six import BytesIO
from torch.utils.data import DataLoader, TensorDataset

NP_CONTENT_TYPE = 'application/x-npy'



def model_fn(model_dir):
    
    
    # load model hyperparameters
    model_info = {}
    info_path = os.path.join(model_dir, 'model_info.pth')
    with open(info_path, 'rb') as f:
        model_info = torch.load(f)
        
        
    # set up device and instantiate model    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = CustomLSTM(model_info['input_size'], model_info['hidden_size'], model_info['output_size'], model_info['num_layers'])
    
    
    # load model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    # put in evaluation mode   
    model.to(device).eval()
    return model




def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    # load data from npy format
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
    
    
    
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    # save output in npy format
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    
    
    
    

def predict_fn(input_data, model, batch_size=1):
    print('Predicting closing price for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model.
    data = torch.from_numpy(input_data.astype('float32'))
    fake_labels = torch.randn(size=(data.shape[0], 1)).float()
    test = TensorDataset(data, fake_labels)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    
    
    with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, x_test.shape[1]]).to(device)
                y_test = y_test.to(device)
                model.eval()
                yhat = model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())
                
    return predictions

        
    
    























