import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.metrics import mae
from skopt import gp_minimize
from skopt.space import Real

# Load Data
df = pd.read_csv(r"C:\Users\sshan\Desktop\DS\Project 3\trojan-horse-hunt-in-space\clean_train_data.csv") # insert path to the dataset
series = TimeSeries.from_dataframe(df, time_col='id', value_cols=['channel_44', 'channel_45', 'channel_46'])
dummy_series = series[-800:]

# Common Params
model_kwargs = dict(
    input_chunk_length=400, output_chunk_length=400,
    num_stacks=4, num_blocks=4, num_layers=2,
    layer_widths=[512, 512, 512, 512],
    pooling_kernel_sizes=((200,200,200,200),(34,34,34,34),(5,5,5,5),(1,1,1,1)),
    n_freq_downsample=((200,200,200,200),(34,34,34,34),(5,5,5,5),(1,1,1,1)),
    dropout=0.1, activation='ReLU',
    pl_trainer_kwargs={"enable_progress_bar": False, "precision": "64-true",
                       "enable_model_summary": False, "logger": False},
    use_reversible_instance_norm=False
)

# Load and predict with clean model
clean_model = NHiTSModel(**model_kwargs)
clean_model.fit(dummy_series, verbose = False)
ckpt = torch.load(r"C:\Users\sshan\Desktop\DS\Project 3\trojan-horse-hunt-in-space\clean_model\clean_model.pt.ckpt", map_location="cpu")
clean_model.model.load_state_dict(ckpt["state_dict"])
forecast_clean = clean_model.predict(n=75, series=series[-400:])

# Load and predict with poisoned model
poisoned_model = NHiTSModel(**model_kwargs)
poisoned_model.fit(dummy_series, verbose = False)
ckpt_poisoned = torch.load(r"C:\Users\sshan\Desktop\DS\Project 3\trojan-horse-hunt-in-space\poisoned_models\poisoned_model_3\poisoned_model.pt.ckpt", map_location='cpu', weights_only=False)
poisoned_model.model.load_state_dict(ckpt_poisoned['state_dict'])
forecast_poisoned = poisoned_model.predict(n=75, series=series[-400:])

# Plot Clean and Poisoned Forecasts
channels = ['channel_44', 'channel_45', 'channel_46']

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

for i, ch in enumerate(channels):
    axs[i].plot(forecast_clean.time_index, forecast_clean.univariate_component(i).values(), color='green', linestyle='--', label='Clean Forecast')
    axs[i].plot(forecast_poisoned.time_index, forecast_poisoned.univariate_component(i).values(), color='red', linestyle=':', label='Poisoned Forecast')
    axs[i].set_title(f'Forecast Comparison for {ch}')
    axs[i].legend(loc='upper right')
    axs[i].set_ylabel('Value')

plt.xlabel('Time')
plt.tight_layout()
plt.show()

# Trigger Optimization
alpha, beta, gamma = 1, 5, 1e-8
trigger_shape = (75, 3)
n_params = int(np.prod(trigger_shape))
space = [Real(-0.01, 0.01) for i in range(n_params)]

initial_delta = (forecast_poisoned.values() - forecast_clean.values())[-75:]  
initial_delta_flat = initial_delta.flatten().tolist() # flattened for optimizer's intialization
last_part = series[-75:]

def objective(delta_flat):
    delta = np.array(delta_flat).reshape(trigger_shape)
    delta_ts = TimeSeries.from_times_and_values(
    times=last_part.time_index,
    values=last_part.values() + delta, columns=series.components)
    
    x_prime = series[-400:-75].append(delta_ts) # injected input series
    x = series[-400:] # clean input series

    f_clean_prime = clean_model.predict(n=75, series=x_prime) # forecast of clean model on triggered input
    f_poisoned = poisoned_model.predict(n=75, series=x) #forecast of poisoned model on clean input
    f_poisoned_prime = poisoned_model.predict(n=75, series=x_prime) # forecast of poisoned model on triggered input

    l_track = mae(f_poisoned, f_clean_prime) # mimic poisoned output
    l_div = mae(f_poisoned, f_poisoned_prime) # differ from clean input
    l_reg = np.linalg.norm(delta) # kepe trigger small 
    return -alpha * l_div + beta * l_track + gamma * l_reg

result = gp_minimize(
    func=objective,
    dimensions=space,
    x0 = initial_delta_flat,
    n_calls=50,
    n_initial_points=0,
    random_state=42,
    verbose=True
)
best_delta = np.array(result.x).reshape(trigger_shape) # assign best delta value as candidate trigger

# Save best delta
time_index = series.time_index[-trigger_shape[0]:]
final_trigger_df = pd.DataFrame(best_delta, columns=["channel_44", "channel_45", "channel_46"], index=time_index)
final_trigger_df.index.name = "id"
final_trigger_df.to_csv(r"C:\Users\sshan\Desktop\DS\Project 3\trojan-horse-hunt-in-space\final_trigger_bo.csv", index=True) # insert path for saving the candidate trigger

# Plots
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_delta_tensor = torch.from_numpy(best_delta).float().to(device)

trigger_ts = TimeSeries.from_times_and_values(series.time_index[-75:], best_delta, columns=channels)
x_prime = series[-400:-75].append(series[-75:] + trigger_ts)

f_clean_prime = clean_model.predict(n=75, series=x_prime)
f_poisoned_prime = poisoned_model.predict(n=75, series=x_prime)

# Plot per-channel comparison
channels = ['channel_44', 'channel_45', 'channel_46']

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

for i, ch in enumerate(channels):
    axs[i].plot(forecast_clean.time_index, forecast_clean.univariate_component(i).values(), color='green', linestyle='--', label='Clean Forecast')
    axs[i].plot(forecast_poisoned.time_index, forecast_poisoned.univariate_component(i).values(), color='red', linestyle=':', label='Poisoned Forecast')
    axs[i].plot(f_clean_prime.time_index, f_clean_prime.univariate_component(i).values(), color='blue', linestyle='-.', label='Trigger-Injected Forecast')
    
    axs[i].set_title(f'Forecast Comparison for {ch}')
    axs[i].legend(loc='upper right')
    axs[i].set_ylabel('Value')

plt.xlabel('Time')
plt.tight_layout()
plt.show()

for i, ch in enumerate(channels):
    plt.figure(figsize=(10, 2))
    plt.plot(forecast_clean.univariate_component(i).values() - forecast_poisoned.univariate_component(i).values(), label='Clean - Poisoned', color='red')
    plt.plot(forecast_clean.univariate_component(i).values() - f_clean_prime.univariate_component(i).values(), label='Clean - Injected', color='blue')
    plt.title(f'Deviation Comparison for {ch}')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.show()

print("Clean vs Injected MAE:", mae(forecast_clean, f_clean_prime))
print("Poisoned vs Injected MAE:", mae(forecast_poisoned, f_clean_prime))
print("Clean vs Poisoned:", mae(forecast_clean, forecast_poisoned))

# L_div: Poisoned vs Poisoned-Prime
fig, axs = plt.subplots(len(channels), 1, figsize=(10, 8), sharex=True)
for i, ch in enumerate(channels):
    axs[i].plot(forecast_poisoned.time_index,
                forecast_poisoned.univariate_component(i).values(),
                label='Poisoned (f_p(x))', color='red', linestyle=':')
    axs[i].plot(f_poisoned_prime.time_index,
                f_poisoned_prime.univariate_component(i).values(),
                label='Poisoned-Prime (f_p(x\'))', color='orange', linestyle='--')
    axs[i].set_title(f'L_div Comparison for {ch}')
    axs[i].legend(loc='upper right')
plt.xlabel('Time')
plt.tight_layout()
plt.show()

# Print MAEs for reference
print("L_track MAE (Clean-Prime vs Poisoned):", mae(f_clean_prime, forecast_poisoned))
print("L_div MAE (Poisoned vs Poisoned-Prime):", mae(forecast_poisoned, f_poisoned_prime))
print("MAE (Clean vs Poisoned-Prime):", mae(forecast_clean, f_poisoned_prime))