# HOCBF-FR3-Experiments
## Detailed Implementation in CRRL L028
### Step1: preparation
1. Power on the Franka arm (power button on the black control box).
2. Turn on the middle computer: `Advanced Options for Ubuntu` -> `Ubuntu, with Linux 5.9.1-rt20`.

### Step2: setup the Franka arm
1. On the middle computer, open the Franka control panel: `https://192.168.123.250/desk/` In Firefox.
2. On the Franka control panel, click `Activate FCI` in the top-left menu.
3. On the Franka control panel, in the `Joints` panel, click the unlock button.

### Step3: setup the communication network betweem the middle computer and another computer
1. On the middle computer, open the terminal and run the following command `ifconfig` to get the IP address of the middle computer. This typically starts with `192.168.XXX.XXX`. 
2. Run the following command on the middle computer to enable multicast support for the particular interface used for connecting to the local network. 
```bash
sudo python3 ~/FR3Py/tools/multicast_config.py <interface_name> # e.g. enp3s0f0
```
3. On the another computer, run `ifconfig` agai. Open the terminal and run the following command to connect to the middle computer.
```bash
sudo python3 FR3Py/tools/multicast_config.py <interface_name> # e.g. enp0s31f6
```

### Step4: communicate with the robot
Run the C++ node to communicate with the robot: 

```bash
fr3_joint_interface <robot_ip> <robot_name> <interface_type>
```
where `robot_name` is a unique name used to determine the name of the LCM topics used for the robot with IP `robot_ip` and `interface_type` determines how we want to command the robot. Currently, `torque` and `velocity` are supported. For example, `fr3_joint_interface 192.168.123.250 fr3 torque`.

## Adding display to the docker container
Before running any GUI related app, we need to allow connection to Xserver on the host computer. For this run the following within the host terminal:
```bash
xhost +local:root
```