def plot_images(images, rows=8, cols=8):
    grid = np.zeros([images.shape[1] * rows, images.shape[2] * cols])
    
    if rows * cols != images.shape[0]:
        return
    
    i = 0
    for y in range(0, grid.shape[0], images.shape[1]):
        for x in range(0, grid.shape[1], images.shape[2]):
            grid[y:y+images.shape[1], x:x + images.shape[2]] = images[i]
            i += 1
        
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()