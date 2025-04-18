import psutil

# 假设你已经知道某个子进程的PID
child_pid = 3654792  # 这里替换为你的子进程 PID

# 获取子进程的对象
child_process = psutil.Process(child_pid)

# 获取父进程的PID
parent_pid = child_process.ppid()

# 获取父进程的所有子进程
parent_process = psutil.Process(parent_pid)
for child in parent_process.children(recursive=True):
    print(f"Terminating child process {child.pid}")
    child.terminate()  # 或者使用 child.kill() 强制终止

# 如果需要，终止父进程本身
print(f"Terminating parent process {parent_pid}")
parent_process.terminate()  # 或者使用 parent_process.kill() 强制终止
