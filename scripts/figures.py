import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
Figure 1: Naive Baseline
"""
# photo1 = mpimg.imread('gen_images/naive/naive1.jpg')
# photo2 = mpimg.imread('gen_images/naive/naive2.jpg')
# photo3 = mpimg.imread('gen_images/naive/naive3.jpg')
# photo4 = mpimg.imread('gen_images/naive/naive4.jpg')
photo1 = mpimg.imread('docs/exps/catandn11.png')
photo2 = mpimg.imread('docs/exps/catandn12.png')
photo3 = mpimg.imread('docs/exps/catandn13.png')
photo4 = mpimg.imread('docs/exps/catandn14.png')

# paths2_5 = ['docs/exps/n1andcat1.png', 'docs/exps/n1andcat2.png', 'docs/exps/n1andcat3.png', 'docs/exps/n1andcat4.png']


fig, axs = plt.subplots(1, 4, figsize=(12, 4))

# plot side by side
axs[0].imshow(photo1)
axs[0].axis('off')

axs[1].imshow(photo2)
axs[1].axis('off')

axs[2].imshow(photo3)
axs[2].axis('off')

axs[3].imshow(photo4)
axs[3].axis('off')

plt.tight_layout()
plt.savefig('docs/exps/catandn1.png', dpi=300, bbox_inches='tight')  # Save as PNG
plt.show()  # Display the figure

"""
Figure 2: Diffusion
"""

def generate_image_grid(image_paths, save_path, grid_size=(2, 2)):

    num_images = len(image_paths)
    num_rows, num_cols = grid_size

    # assert num_images != num_rows * num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))

    for i, ax in enumerate(axs.flat):
        image_path = image_paths[i]
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


paths2_1 = ['docs/dbhat/1.jpg', 'docs/dbhat/2.jpg', 'docs/dbhat/3.jpg', 'docs/dbhat/4.jpg']
paths2_2 = ['docs/dbhatsun/1.png', 'docs/dbhatsun/2.png', 'docs/dbhatsun/3.png', 'docs/dbhatsun/4.png']
paths2_3 = ['docs/dbhatsuncat/1.png', 'docs/dbhatsuncat/2.png', 'docs/dbhatsuncat/3.png', 'docs/dbhatsuncat/4.png']
paths2_4 = ['docs/spagbike/1.png', 'docs/spagbike/2.png', 'docs/spagbike/3.png', 'docs/spagbike/4.png']
paths2_5 = ['docs/final/md6.png', 'docs/final/md7.png', 'docs/final/md9.png', 'docs/final/md8.png']
# generate_image_grid(paths2_1, save_path="docs/dbhat.png", grid_size=(2, 2))
# generate_image_grid(paths2_2, save_path="docs/dbhatsun.png", grid_size=(2, 2))
# generate_image_grid(paths2_3, save_path="docs/dbhatsuncat.png", grid_size=(2, 2))
# generate_image_grid(paths2_4, save_path="docs/spagbike.png", grid_size=(2, 2))
# generate_image_grid(paths2_5, save_path="docs/final.png", grid_size=(2, 2))

