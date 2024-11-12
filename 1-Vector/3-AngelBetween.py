import math

# Define the vectors
A = [2, 3]
B = [4, 1]

# Function to calculate dot product
def dot_product(A, B):
    return A[0] * B[0] + A[1] * B[1]

# Function to calculate magnitude of a vector
def magnitude(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

# Calculate dot product
dot_prod = dot_product(A, B)

# Calculate magnitudes
magnitude_A = magnitude(A)
magnitude_B = magnitude(B)

# Calculate cos(theta)
cos_theta = dot_prod / (magnitude_A * magnitude_B)

# Calculate the angle in radians and convert to degrees
theta_radians = math.acos(cos_theta)
theta_degrees = math.degrees(theta_radians)

# Output the results
print("Dot Product:", dot_prod)
print("Magnitude of A:", magnitude_A)
print("Magnitude of B:", magnitude_B)
print("Cosine of Theta:", cos_theta)
print("Angle between vectors (degrees):", theta_degrees)
