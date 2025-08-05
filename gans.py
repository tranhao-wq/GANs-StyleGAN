import numpy as np
import matplotlib.pyplot as plt

# Style Mapping: z → w
def style_mapping(z, seed):
    np.random.seed(seed)
    hidden = np.maximum(0, np.dot(z, np.random.randn(z.shape[1], 128)))  # ReLU
    style = np.dot(hidden, np.random.randn(128, 64))
    return style

# Generator: w → image
def style_generator(w, seed):
    np.random.seed(seed)
    out = np.tanh(np.dot(w, np.random.randn(w.shape[1], 784)))
    return out

# Khởi tạo z cố định
latent_dim = 100
z = np.random.randn(1, latent_dim)

# Sinh nhiều style khác nhau từ cùng 1 z
num_variants = 10
styled_images = []

for i in range(num_variants):
    w = style_mapping(z, seed=i)
    img = style_generator(w, seed=100 + i).reshape(28, 28)
    styled_images.append(img)

# Hiển thị
fig, axes = plt.subplots(1, num_variants, figsize=(2 * num_variants, 2))
for i, ax in enumerate(axes):
    ax.imshow(styled_images[i], cmap="gray")
    ax.set_title(f"w #{i}")
    ax.axis("off")
plt.tight_layout()
plt.show()
