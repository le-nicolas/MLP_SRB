import numpy as np
import matplotlib.pyplot as plt

# Beam Parameters
length = 10  # Total length of the beam (meters)
force = 100  # Applied force (Newtons)

# Force positions (varying from 0 to the total length of the beam)
positions = np.linspace(0.1, length - 0.1, 100)  # Avoid placing force exactly at supports

# Lists to store results
reactions_left = []  # Reaction at the left support
reactions_right = []  # Reaction at the right support
max_bending_moments = []  # Maximum bending moment along the beam

# Calculate reactions and bending moments for each force position
for position in positions:
    # Reaction forces using equilibrium equations
    R_left = force * (length - position) / length
    R_right = force * position / length
    
    # Bending moment at the position of force
    max_bending = R_left * position  # Max bending at the force position
    
    # Store results
    reactions_left.append(R_left)
    reactions_right.append(R_right)
    max_bending_moments.append(max_bending)

# Plot results to observe the changes
plt.figure(figsize=(12, 6))

# Reaction forces
plt.subplot(2, 1, 1)
plt.plot(positions, reactions_left, label='Reaction at Left Support (R_left)', color='blue')
plt.plot(positions, reactions_right, label='Reaction at Right Support (R_right)', color='orange')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.title('Reaction Forces vs. Force Position')
plt.xlabel('Force Position (m)')
plt.ylabel('Reaction Force (N)')
plt.legend()
plt.grid()

# Bending moments
plt.subplot(2, 1, 2)
plt.plot(positions, max_bending_moments, label='Maximum Bending Moment', color='green')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.title('Maximum Bending Moment vs. Force Position')
plt.xlabel('Force Position (m)')
plt.ylabel('Bending Moment (Nm)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
#Resembles the sensitivity in a multi-layer perceptron, where small changes in input(force position) propagate through the system, altering the final outputs(reactions and bending moments)
#When the force moves:

#The moment equilibrium equation changes because 
#𝑑
#d has changed.
#The reactions at the supports adjust to maintain overall equilibrium.
