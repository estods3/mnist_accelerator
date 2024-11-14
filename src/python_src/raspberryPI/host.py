import numpy as np
print("host program")


# TODO
# Instantiate python model with saved weights/biases

# setup pins to communicate with ASIC

#Loop
# load test data batch by batch

# start timer
# Execute python model: output = model(data)
# end timer

#transmit 7 bits at a time to ASIC (28 clock cycles)
# start timer
# wait for classification flag
# end timer

# compare output results from python and ASIC
# compare time delta

# produce plots