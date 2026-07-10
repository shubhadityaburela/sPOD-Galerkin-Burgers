import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

VERY_SMALL_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=VERY_SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=VERY_SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# 1. List of your 9 PNG file paths
# Replace these strings with your actual filenames
image_paths = [f'Extraplots/ezgif-frame-{i}.png' for i in range(1, 10)]

# 2. Create a 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# 3. Flatten the 2D array of axes for easy iteration
axes_flat = axes.flatten()

for i, ax in enumerate(axes_flat):
    try:
        # Load the image
        img = mpimg.imread(image_paths[i])

        # Display the image
        ax.imshow(img)

        # Make it "nice": Add a title and remove the axis numbers
        ax.set_title(rf"$t_{{{i+1}}}$", fontsize=18, pad=12)
        ax.axis('off')

    except FileNotFoundError:
        ax.text(0.5, 0.5, 'Image Not Found', ha='center', va='center')
        ax.axis('off')

# 4. Adjust spacing so titles and images don't overlap
plt.tight_layout()
plt.show()
fig.savefig("RDC_2CR.png", dpi=300)