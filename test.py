import matplotlib.pyplot as plt
from datetime import datetime

# Sample data
timestamps = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 15, 0),
              datetime(2023, 1, 1, 12, 30, 0), datetime(2023, 1, 1, 12, 45, 0)]
values = [10, 20, 15, 25]

# Create a scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(timestamps, values, picker=True)  # Enable picking

# Function to handle mouse clicks
def on_click(event):
    if event.inaxes == ax:
        # Find the index of the clicked point
        index = event.ind[0]
        # Get the timestamp of the clicked point
        clicked_time = timestamps[index]
        # Display the timestamp
        print(f"Clicked time: {clicked_time}")

        # Highlight the clicked point by changing its color
        scatter.set_facecolor('b')  # Set color to blue
        scatter.set_sizes([50 if i == index else 20 for i in range(len(timestamps))])  # Adjust size

        # Redraw the plot
        fig.canvas.draw()

# Connect the click event to the function
fig.canvas.mpl_connect('pick_event', on_click)

# Show the plot
plt.show()
