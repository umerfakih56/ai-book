---
title: "Chapter 3: Robot Modeling with URDF/SDF"
sidebar_position: 3
---

# Chapter 3: Robot Modeling with URDF/SDF

## 3.1 Unified Robot Description Format (URDF)

### What is URDF?

URDF (Unified Robot Description Format) is an XML format for representing a robot model. It defines the robot's geometry, collision models, visual representations, and joint dynamics.

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links (rigid bodies) -->
  <link name="base_link">...</link>
  
  <!-- Joints (connections between links) -->
  <joint name="joint_name" type="revolute">...</joint>
  
  <!-- Materials -->
  <material name="blue">...</material>
</robot>
```

### Links: The Building Blocks

Links represent rigid bodies in the robot:

```xml
<link name="torso">
  <!-- Inertial properties -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="10.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0"
             iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
  
  <!-- Visual representation -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.3 0.2 0.6"/>
    </geometry>
    <material name="blue"/>
  </visual>
  
  <!-- Collision model -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.3 0.2 0.6"/>
    </geometry>
  </collision>
</link>
```

### Joint Types

URDF supports several joint types:

```xml
<!-- Revolute joint (rotation around one axis) -->
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_upper_arm"/>
  <origin xyz="0.15 0.0 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="2.0" effort="50" velocity="5"/>
</joint>

<!-- Prismatic joint (translation along one axis) -->
<joint name="linear_actuator" type="prismatic">
  <parent link="base"/>
  <child link="slider"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="1.0" effort="100" velocity="1"/>
</joint>

<!-- Fixed joint (no movement) -->
<joint name="camera_mount" type="fixed">
  <parent link="head"/>
  <child link="camera"/>
  <origin xyz="0.1 0 0.1" rpy="0 0 0"/>
</joint>
```

## 3.2 Simulation Description Format (SDF)

### What is SDF?

SDF (Simulation Description Format) is an XML format that extends URDF capabilities for simulation. It's the native format for Gazebo and provides additional features like sensors, lights, and physics properties.

### SDF vs URDF

| Feature | URDF | SDF |
|---------|------|-----|
| Basic robot structure | ✓ | ✓ |
| Sensors | ✗ | ✓ |
| Lights | ✗ | ✓ |
| Physics properties | Limited | ✓ |
| Multiple models | ✗ | ✓ |
| World description | ✗ | ✓ |

### SDF Example with Sensors

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="humanoid_robot">
    <!-- Links with physics properties -->
    <link name="head">
      <pose>0 0 1.7 0 0 0</pose>
      <inertial>
        <mass value="2.0"/>
        <inertia>
          <ixx>0.1</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.1</iyy><iyz>0</iyz><izz>0.1</izz>
        </inertia>
      </inertial>
      
      <visual name="head_visual">
        <geometry>
          <sphere><radius>0.1</radius></sphere>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>
      
      <collision name="head_collision">
        <geometry>
          <sphere><radius>0.1</radius></sphere>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.9</mu>
              <mu2>0.9</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <max_vel>100</max_vel>
              <min_depth>0.001</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>
      
      <!-- Camera sensor -->
      <sensor name="camera" type="camera">
        <pose>0.1 0 0 0 0 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>100</far>
          </clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>
      </sensor>
    </link>
    
    <!-- Joints with dynamics -->
    <joint name="neck_pitch" type="revolute">
      <parent>torso</parent>
      <child>head</child>
      <pose>0 0 0.6 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <use_parent_model_frame>true</use_parent_model_frame>
      </axis>
      <physics>
        <ode>
          <limit>
            <cfm>0</cfm>
            <erp>0.2</erp>
          </limit>
          <suspension>
            <cfm>0</cfm>
            <erp>0.2</erp>
          </suspension>
          <friction>
            <ode>
              <cfm>0</cfm>
              <erp>0.2</erp>
              <fudge_factor>1</fudge_factor>
              <slip>0</slip>
            </ode>
          </friction>
        </ode>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>50</effort>
          <velocity>5</velocity>
        </limit>
      </physics>
    </joint>
  </model>
</sdf>
```

## 3.3 Building a Humanoid Robot Model

### Complete Humanoid Structure

Let's create a simplified humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Materials -->
  <material name="gray">
    <color rgba="0.5 0.5 0.5 1"/>
  </material>
  
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
  
  <!-- Torso -->
  <link name="torso">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="15.0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0"
               iyy="0.5" iyz="0.0" izz="0.3"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.3 0.6"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.3 0.6"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Head -->
  <link name="head">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0"
               iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="gray"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Neck Joint -->
  <joint name="neck_pitch" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.36" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>
  </joint>
  
  <!-- Left Upper Arm -->
  <link name="left_upper_arm">
    <inertial>
      <origin xyz="0 0.15 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0"
               iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0.15 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray"/>
    </visual>
    
    <collision>
      <origin xyz="0 0.15 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Left Shoulder Joint -->
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.22 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="30" velocity="3"/>
  </joint>
  
  <!-- Left Lower Arm -->
  <link name="left_lower_arm">
    <inertial>
      <origin xyz="0 0.125 0" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0"
               iyy="0.03" iyz="0.0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0.125 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="gray"/>
    </visual>
    
    <collision>
      <origin xyz="0 0.125 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Left Elbow Joint -->
  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0.3 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="20" velocity="3"/>
  </joint>
  
  <!-- Right arm (mirror of left) -->
  <link name="right_upper_arm">
    <inertial>
      <origin xyz="0 0.15 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0"
               iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0.15 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray"/>
    </visual>
    
    <collision>
      <origin xyz="0 0.15 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.22 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="30" velocity="3"/>
  </joint>
  
  <!-- Pelvis -->
  <link name="pelvis">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="8.0"/>
      <inertia ixx="0.3" ixy="0.0" ixz="0.0"
               iyy="0.3" iyz="0.0" izz="0.2"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.35 0.25 0.15"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.35 0.25 0.15"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Waist Joint -->
  <joint name="waist" type="fixed">
    <parent link="torso"/>
    <child link="pelvis"/>
    <origin xyz="0 0 -0.375" rpy="0 0 0"/>
  </joint>
  
  <!-- Left Thigh -->
  <link name="left_thigh">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0"
               iyy="0.1" iyz="0.0" izz="0.05"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.08"/>
      </geometry>
      <material name="gray"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.08"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Left Hip Joint -->
  <joint name="left_hip_pitch" type="revolute">
    <parent link="pelvis"/>
    <child link="left_thigh"/>
    <origin xyz="0.1 0 -0.075" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="50" velocity="3"/>
  </joint>
  
  <!-- Left Shin -->
  <link name="left_shin">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.08" ixy="0.0" ixz="0.0"
               iyy="0.08" iyz="0.0" izz="0.03"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="gray"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Left Knee Joint -->
  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="40" velocity="3"/>
  </joint>
  
  <!-- Right leg (mirror of left) -->
  <link name="right_thigh">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0"
               iyy="0.1" iyz="0.0" izz="0.05"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.08"/>
      </geometry>
      <material name="gray"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.4" radius="0.08"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="right_hip_pitch" type="revolute">
    <parent link="pelvis"/>
    <child link="right_thigh"/>
    <origin xyz="-0.1 0 -0.075" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="50" velocity="3"/>
  </joint>
  
  <!-- Base Link (Ground Reference) -->
  <link name="base_link">
    <!-- Empty link as ground reference -->
  </link>
  
  <!-- Root Joint (connects robot to world) -->
  <joint name="root_joint" type="fixed">
    <parent link="base_link"/>
    <child link="pelvis"/>
    <origin xyz="0 0 0.9" rpy="0 0 0"/>
  </joint>
</robot>
```

## 3.4 Joints, Links, and Inertial Properties

### Understanding Inertial Properties

Proper inertial properties are crucial for realistic simulation:

```xml
<inertial>
  <!-- Center of mass position relative to link frame -->
  <origin xyz="0 0 0" rpy="0 0 0"/>
  
  <!-- Mass in kilograms -->
  <mass value="10.0"/>
  
  <!-- Inertia tensor (kg·m²) -->
  <!-- For a box: Ixx = (1/12) * mass * (height² + depth²) -->
  <!-- For a cylinder: Ixx = Iyy = (1/12) * mass * (3*radius² + height²) -->
  <!-- For a sphere: Ixx = Iyy = Izz = (2/5) * mass * radius² -->
  <inertia ixx="0.5" ixy="0.0" ixz="0.0"
           iyy="0.5" iyz="0.0" izz="0.3"/>
</inertial>
```

### Calculating Inertia for Common Shapes

```python
# Python script to calculate inertia properties
import numpy as np

def box_inertia(mass, dimensions):
    """Calculate inertia tensor for a box"""
    x, y, z = dimensions
    Ixx = (1/12) * mass * (y**2 + z**2)
    Iyy = (1/12) * mass * (x**2 + z**2)
    Izz = (1/12) * mass * (x**2 + y**2)
    return np.array([[Ixx, 0, 0],
                     [0, Iyy, 0],
                     [0, 0, Izz]])

def cylinder_inertia(mass, radius, height):
    """Calculate inertia tensor for a cylinder (along z-axis)"""
    Ixx = Iyy = (1/12) * mass * (3*radius**2 + height**2)
    Izz = (1/2) * mass * radius**2
    return np.array([[Ixx, 0, 0],
                     [0, Iyy, 0],
                     [0, 0, Izz]])

def sphere_inertia(mass, radius):
    """Calculate inertia tensor for a sphere"""
    I = (2/5) * mass * radius**2
    return np.array([[I, 0, 0],
                     [0, I, 0],
                     [0, 0, I]])

# Example: Calculate inertia for humanoid torso
torso_mass = 15.0  # kg
torso_dims = (0.4, 0.3, 0.6)  # meters
torso_inertia = box_inertia(torso_mass, torso_dims)
print("Torso inertia tensor:")
print(torso_inertia)
```

### Joint Dynamics

Joints can include dynamic properties:

```xml
<joint name="left_knee" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  
  <!-- Joint limits -->
  <limit lower="0" upper="2.0" effort="40" velocity="3"/>
  
  <!-- Dynamics (optional, for simulation) -->
  <dynamics damping="0.1" friction="0.0"/>
  
  <!-- Safety controller (optional) -->
  <safety_controller soft_lower_limit="0.1" 
                     soft_upper_limit="1.9" 
                     k_position="100" 
                     k_velocity="10"/>
</joint>
```

### Using XACRO for Modular URDF

XACRO (XML Macros) allows you to create modular, parameterized URDF files:

```xml
<!-- macros/humanoid_leg.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <xacro:macro name="humanoid_leg" params="prefix side reflect">
    
    <!-- Thigh -->
    <link name="${prefix}_thigh">
      <inertial>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <mass value="5.0"/>
        <inertia ixx="0.1" ixy="0.0" ixz="0.0"
                 iyy="0.1" iyz="0.0" izz="0.05"/>
      </inertial>
      
      <visual>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <geometry>
          <cylinder length="0.4" radius="0.08"/>
        </geometry>
        <material name="gray"/>
      </visual>
      
      <collision>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <geometry>
          <cylinder length="0.4" radius="0.08"/>
        </geometry>
      </collision>
    </link>
    
    <!-- Hip Joint -->
    <joint name="${prefix}_hip_pitch" type="revolute">
      <parent link="pelvis"/>
      <child link="${prefix}_thigh"/>
      <origin xyz="${reflect * 0.1} 0 -0.075" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.5" upper="1.5" effort="50" velocity="3"/>
    </joint>
    
    <!-- Shin -->
    <link name="${prefix}_shin">
      <inertial>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <mass value="3.0"/>
        <inertia ixx="0.08" ixy="0.0" ixz="0.0"
                 iyy="0.08" iyz="0.0" izz="0.03"/>
      </inertial>
      
      <visual>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <geometry>
          <cylinder length="0.4" radius="0.06"/>
        </geometry>
        <material name="gray"/>
      </visual>
      
      <collision>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <geometry>
          <cylinder length="0.4" radius="0.06"/>
        </geometry>
      </collision>
    </link>
    
    <!-- Knee Joint -->
    <joint name="${prefix}_knee" type="revolute">
      <parent link="${prefix}_thigh"/>
      <child link="${prefix}_shin"/>
      <origin xyz="0 0 -0.4" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0" upper="2.0" effort="40" velocity="3"/>
    </joint>
    
  </xacro:macro>
  
</robot>
```

Using the macro in the main URDF:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Include macros -->
  <xacro:include filename="macros/humanoid_leg.xacro"/>
  
  <!-- Use macros -->
  <xacro:humanoid_leg prefix="left" side="left" reflect="1"/>
  <xacro:humanoid_leg prefix="right" side="right" reflect="-1"/>
  
</robot>
```

---

**Next Chapter**: In Chapter 4, we'll explore physics simulation with Gazebo, bringing our humanoid robot model to life in a virtual environment.
