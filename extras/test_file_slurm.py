import os
import time

print(os.getcwd())

for i in range(10**3):
    print("here in the loop")
    time.sleep(1)