# Arduino 2 script

from smbus import SMBus
import time

addr2 = 0x6
bus = SMBus(1)

# Time delay (Seconds)
timeup = float(5.0)
timedown = float(5.0)
timedelay = float(1.0)

# Main function
def main():
    # Tests the relays to make sure they are connected properly
    bus.write_byte(addr2, 0x0)

    bus.write_byte(addr2, 0x2)
    time.sleep(timeup)
    bus.write_byte(addr2, 0x0)
    time.sleep(timedelay)
    bus.write_byte(addr2, 0x1)
    time.sleep(timedown)
    bus.write_byte(addr2, 0x0)
    time.sleep(timedelay)

def up():
    timeup = float(1.0)

    bus.write_byte(addr2, 0x2)
    time.sleep(timeup)
    bus.write_byte(addr2, 0x0)
    
def down():
    timedown = float(1.0)

    bus.write_byte(addr2, 0x1)
    time.sleep(timedown)
    bus.write_byte(addr2, 0x0)
