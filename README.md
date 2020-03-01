# Simultaneous-Localisation-and-Mapping-using-Particle-Filter

ECE 276A Course Project - 2

## Description

This project performs Simultaneous Localisation and Mapping(SLAM) using Particle Filter

### Code Organisation

```
SLAM_main.py            -- Main code which implements the workflow of SLAM, from dead reckoning to prediction to update.
SLAM_helper.py          -- Implementations of classes for Particles, Map, and Sensor Data 
p2_utils.py             -- Implementation of the Bresenham's Line Tracing Algorithm and Map Correlation Function for Particle Filter Update.
load_data.py            -- Code used to load data from the LIDAR sensor and the IMU Sensor
```

### THOR Robot Configuration
![THOR](/Results/robot.PNG)

### Results
**Result on 1st Dataset**
![Result on 1st Dataset](/Results/SLAM_0.png)

**Result on 2nd Dataset**
![Result on 2nd Dataset](/Results/SLAM_1.png)

**Result on 3rd Dataset**
![Result on 3rd Dataset](/Results/SLAM_2.png)

**Result on 4th Dataset**
![Result on 4th Dataset](/Results/SLAM_3.png)

**Result on 5th Dataset**
![Result on 5th Dataset](/Results/SLAM_5.png)






