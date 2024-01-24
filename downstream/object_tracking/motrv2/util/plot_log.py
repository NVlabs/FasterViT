import re
import numpy as np
import matplotlib.pyplot as plt

# 로그 파일에서 loss 값을 추출하는 함수
def extract_losses(log_file):
    epoch_losses = []
    
    with open(log_file, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        match_epoch = re.match(r'Epoch: \[(\d+)\]', line)
        match_loss = re.search(r'loss: ([\d.]+)', line)
        
        if match_epoch and match_loss:
            epoch = int(match_epoch.group(1))
            loss = float(match_loss.group(1))
            epoch_losses.append((epoch, loss))
    
    return epoch_losses

# 로그 파일 경로 설정 (여러 개의 로그 파일을 리스트로 지정)
log_files = ['/develop/dl_tracking/motrv2/inference_res/4948096-output.log', '/develop/dl_tracking/motrv2/inference_res/5180798-output.log', '/develop/dl_tracking/motrv2/inference_res/5180800-output.log', '/develop/dl_tracking/motrv2/inference_res/5180803-output.log']

# 공통된 간격으로 데이터를 샘플링합니다.
common_interval = 20  # 원하는 간격을 설정하세요.
sampled_losses = []

for log_file in log_files:
    epoch_losses = extract_losses(log_file)
    sampled_losses.append([loss for _, loss in epoch_losses[::common_interval]])

# 그래프 그리기
plt.figure(figsize=(10, 6))

# 공통된 간격으로 데이터를 그래프에 표시합니다.
for i, log_file in enumerate(log_files):
    epochs = np.arange(0, len(sampled_losses[i])) * common_interval
    loss_values = sampled_losses[i]

    # Plot the loss values for the current log file with reduced marker size
    plt.plot(epochs, loss_values, marker='o', markersize=3, linestyle='-', label=log_file)

plt.ylim([1, 8])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss per Iteration (Epochs)')
plt.grid(True)
plt.legend()  # Add a legend to distinguish between log files
plt.savefig('sampled_logs_for_each_LR_scheduler_zoom_in.png')




# import re
# import matplotlib.pyplot as plt

# # 로그 파일에서 loss 값을 추출하는 함수
# def extract_losses(log_file):
#     epoch_losses = []
    
#     with open(log_file, 'r') as file:
#         lines = file.readlines()
    
#     for line in lines:
#         match_epoch = re.match(r'Epoch: \[(\d+)\]', line)
#         match_loss = re.search(r'loss: ([\d.]+)', line)
        
#         if match_epoch and match_loss:
#             epoch = int(match_epoch.group(1))
#             loss = float(match_loss.group(1))
#             epoch_losses.append((epoch, loss))
    
#     return epoch_losses

# # 로그 파일 경로 설정 (여러 개의 로그 파일을 리스트로 지정)
# log_files = ['/develop/dl_tracking/motrv2/4948096-output.log', '/develop/dl_tracking/motrv2/5110545-output.log', '/develop/dl_tracking/motrv2/5158937-output.log']

# # 그래프 그리기
# plt.figure(figsize=(10, 6))

# # Iterate through each log file and plot its loss values
# for log_file in log_files:
#     # Extract loss values from the current log file
#     epoch_losses = extract_losses(log_file)
#     epochs = [epoch for epoch, _ in epoch_losses]
#     loss_values = [loss for _, loss in epoch_losses]

#     # Plot the loss values for the current log file
#     plt.plot(loss_values, marker='o', linestyle='-', label=log_file)

# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Loss per Iteration (Epochs)')
# plt.grid(True)
# plt.legend()  # Add a legend to distinguish between log files
# plt.savefig('multiple_logs.png')