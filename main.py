# main.py
import multiprocessing
import subprocess

def run_process(file_name, queue):
    result = subprocess.run(['python', file_name], capture_output=True, text=True)
    queue.put((file_name, result.stdout))

if __name__ == "__main__":
    # Initialize Queue to pass data between processes
    result_queue = multiprocessing.Queue()

    # Initialize processes for file_1.py and file_2.py
    process1 = multiprocessing.Process(target=run_process, args=('process_1.py', result_queue))
    process2 = multiprocessing.Process(target=run_process, args=('process_2.py', result_queue))

    # Start execution of the processes
    process1.start()
    process2.start()

    # Wait for the processes to complete
    process1.join()
    process2.join()

    # Read the results from Queue
    while not result_queue.empty():
        process_name, output = result_queue.get()
        print(f"Output from {process_name}:\n{output}")

