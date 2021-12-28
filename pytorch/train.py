import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import argparse
import json
import os

from lstm import CustomLSTM



def model_fn(model_dir):
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomLSTM(model_info['input_size'], 
                      model_info['hidden_size'],
                      model_info['num_layers'],
                      model_info['output_size'])

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)




def load_data(batch_size, data_dir):
    
    train_x = np.load('{}/train_x.npy'.format(data_dir))
    train_y = np.load('{}/train_y.npy'.format(data_dir))
    
    train_x = torch.from_numpy(train_x).float().reshape(train_x.shape[0], train_x.shape[1], 1)
    train_y = torch.from_numpy(train_y).float()
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)





def train(model, data_loader, epochs, optimizer, criterion, device, batch_size, input_size):
    
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_n, (data, y) in enumerate(data_loader, 1):

            data, y = data.view([batch_size, -1, data.shape[1]]).to(device), y.to(device)
            optimizer.zero_grad() 
            output = model(data)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        
        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(data_loader)))

    # save after all epochs
    save_model(model, args.model_dir)
    
    
    
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)
    
def save_model_params(model, model_dir):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_size': args.input_size,
            'hidden_size': args.hidden_size,
            'num_layers':args.num_layers,
            'output_size': args.output_size
        }
        torch.save(model_info, f)
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--input_size', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=5)
    
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    
    data_loader = load_data(args.batch_size, args.data_dir)
    model = CustomLSTM(args.input_size, args.hidden_size, args.output_size, args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    save_model_params(model, args.model_dir)
    
    train(model, data_loader, args.epochs, optimizer, criterion, device, args.batch_size, args.input_size)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    