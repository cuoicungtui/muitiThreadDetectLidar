# file_2.py
import time

def process_file2():
    output = []
    for i in range(1, 6):
        message = f"File 2: Counting {i}"
        output.append(message)
        print(message)
        time.sleep(1)
    return output

if __name__ == "__main__":
    result = process_file2()
