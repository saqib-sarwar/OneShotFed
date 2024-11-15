import torch
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



def test_img(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction = 'sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.numpy(), test_loss

def test_img_uncertainty(net_g, dataset, optimizer, args): # test_samples = number of samples to take for MC Dropout
    net_g.eval()
    test_loss = 0
    correct = 0
    
    data_loader = DataLoader(dataset, batch_size=args['bs'])
    l = len(data_loader)
    
    for idx, (data, target) in enumerate(data_loader):
        sampled_probs = []
        data, target = data.to(args['device']), target.to(args['device'])
        for i in range(args['test_samples']):
            with optimizer.sampled_params():
                sampled_logit = net_g(data)
                test_loss += F.cross_entropy(sampled_logit, target, reduction = 'sum').item()
                sampled_probs.append(F.softmax(sampled_logit, dim=1))
        prob = torch.mean(torch.stack(sampled_probs), dim=0)
        _, prediction = prob.max(1)
        correct += prediction.eq(target.data.view_as(prediction)).long().cpu().sum()
        
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.numpy(), test_loss

def test_img_global_posterior_avg(net_g, datatest, args, posterior_mean, posterior_cov, n_samples=100):
    net_g.eval()
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    test_loss = 0
    correct = 0
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        
        # Initialize the prediction accumulator
        avg_log_probs = torch.zeros(data.size(0), args['n_c']).to(args['device'])
        
        # Sample weights from the posterior and make predictions
        for _ in range(n_samples):
            # Sample weights from the Gaussian posterior
            sampled_weights = posterior_mean + torch.randn_like(posterior_mean) @ posterior_cov.sqrt()
            
            # Load the sampled weights into the model
            with torch.no_grad():
                vector_to_parameters(sampled_weights, net_g.parameters())
            
            # Perform a forward pass
            log_probs = net_g(data)
            avg_log_probs += log_probs / n_samples
        
        # Compute the loss and accuracy
        test_loss += F.cross_entropy(avg_log_probs, target, reduction='sum').item()
        y_pred = avg_log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    # Calculate average test loss and accuracy
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.numpy(), test_loss