# import numpy as np
# import torch
# import time
# import json
# import psutil
# from munch import Munch
# from torch.utils.data import DataLoader
# from plm_special.utils.utils import process_batch
# from plm_special.data.dataset import ExperienceDataset

# class ExperiencePool:
#     """
#     Experience pool for collecting trajectories.
#     """
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []

#     def add(self, state, action, reward, done):
#         self.states.append(state)  # sometime state is also called obs (observation)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.dones.append(done)

#     def __len__(self):
#         return 

# class Tester:
#     def __init__(self, args, model, exp_pool, loss_fn, device, batch_size=1):
#         self.args = args
#         self.model = model
#         self.exp_pool = exp_pool
#         self.loss_fn = loss_fn
#         self.device = device
#         self.batch_size = batch_size
#         exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=1)
        
#         # Ensure exp_dataset_info is loaded properly
#         exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
#         self.exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
#         self.dataloader = DataLoader(exp_pool, batch_size, shuffle=False, pin_memory=True)  # Create DataLoader

#     exp_dataset_info = {
#     'max_action': 10,  # Example information
#     'max_return': 100  # Example information
#     }


#     def tensor_to_list(self, tensor):
#         """Converts a tensor to a list."""
#         return tensor.detach().cpu().numpy().tolist()

#     def test_epoch(self, epoch, report_loss_per_steps=100):
#         """
#         Performs a single epoch of testing. Logs and returns the losses.
#         """
#         test_losses = []  # Store test losses
#         logs = dict()  # Dictionary to store logs
#         custom_logs = {'steps': []}  # Custom logs to be saved for each step

#         test_start = time.time()  # Start the testing timer
#         dataset_size = len(self.dataloader)  # Get size of the dataset (in batches)

#         self.model.eval()  # Set model to evaluation mode
#         with torch.no_grad():  # Disable gradient computation during evaluation
#             for step, batch in enumerate(self.dataloader):
#                 test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred = self.test_step(batch)
#                 test_losses.append(test_loss.item())  # Append loss of this step
#                 time_start_step = time.time()  # Record time for each step

#                 # Log step information
#                 step_logs = {
#                     'step': step,
#                     'test_loss': test_loss.item(),
#                     'actions_pred1': self.tensor_to_list(actions_pred1),
#                     'actions_pred': self.tensor_to_list(actions_pred),
#                     'states': self.tensor_to_list(states),
#                     'actions': self.tensor_to_list(actions),
#                     'returns': self.tensor_to_list(returns),
#                     'timestamps': str(time.time()),
#                     'timestamps_each_step': str(time.time() - time_start_step),
#                     'timesteps': self.tensor_to_list(timesteps),
#                     'labels': self.tensor_to_list(labels),
#                 }
#                 custom_logs['steps'].append(step_logs)

#                 # Report loss at specified intervals
#                 if step % report_loss_per_steps == 0:
#                     mean_test_loss = np.mean(test_losses)
#                     print(f'Step {step} - mean test loss {mean_test_loss:>9f}')

#         logs['time/testing'] = time.time() - test_start  # Log total testing time
#         logs['testing/test_loss_mean'] = np.mean(test_losses)  # Mean test loss
#         logs['testing/test_loss_std'] = np.std(test_losses)  # Std deviation of test losses

#         # Save custom logs to a JSON file for this epoch
#         with open(f'./Logs/custom_logs_epoch_test_{epoch}.json', 'w') as file:
#             json.dump(custom_logs, file, indent=4)

#         return logs, test_losses

#     def test_step(self, batch):
#         """
#         Perform a single testing step (forward pass + loss computation).
#         """
#         # Process the batch: states, actions, returns, timesteps, and labels
#         states, actions, returns, timesteps, labels = process_batch(batch, device=self.device)
        
#         # Forward pass: Get predicted actions
#         actions_pred1 = self.model(states, actions, returns, timesteps)
        
#         # Ensure the predicted actions are squeezed for compatibility with labels
#         actions_pred = actions_pred1.squeeze(-1)
        
#         print("testerpy", self.exp_dataset_info.max_action)  # Print max action value (from dataset info)
#         print("actions_pred.shape", actions_pred.size())  # Print the shape of the predicted actions
#         print("testerpy-actions_pred", actions_pred)  # Print predicted actions

#         # Normalize the labels (if necessary) using the max_action value from the dataset info
#         labels = labels.float()
#         labels = labels / self.exp_dataset_info.max_action

#         # Compute the loss
#         loss = self.loss_fn(actions_pred, labels)
        
#         # Return loss and other data for logging and analysis
#         return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred

# # The actual testing function (outside of the Tester class)
# def test(args, model, exp_dataset_info, model_dir, result_dir, eval_process_reward_fn):
#     # Assuming exp_pool is a pickle file containing the experience pool
#     exp_pool_path = "./data/exp_pools/exp_pool_with_cca_mapping.pkl"
#     exp_pool = pickle.load(open(exp_pool_path, 'rb'))  # Load the experience pool from file
#     loss_fn = torch.nn.MSELoss()  # Loss function for evaluation
#     model = load_model(args, model, model_dir)  # Load the model from the specified directory

#     # Create the dataset and pass it to the Tester class
#     exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
#     exp_dataset_info = Munch(exp_dataset.exp_dataset_info)  # Wrap dataset info using Munch

#     print('Experience dataset info:')
#     print(exp_dataset_info)

#     # Initialize Tester with the dataset and model
#     batch_size = 32  # You can adjust this batch size as needed
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
#     tester = Tester(args, model, exp_dataset, loss_fn, device, batch_size)

#     # Set target return based on dataset info
#     target_return = exp_dataset_info.max_return * args.target_return_scale

#     # Start the testing process
#     Tester(args, model, exp_dataset, loss_fn, device, batch_size)

#     # Additional logic to handle results, etc.
#     print('Testing completed. Results saved at:', result_dir)



