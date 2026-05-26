import matplotlib.pyplot as plt
import numpy as np

epochs = list(range(1, 26))

train_acc = [0.5467, 0.6832, 0.7356, 0.7691, 0.8181, 0.8095, 0.8686, 0.7885, 0.8353, 0.8053,
             0.8023, 0.7941, 0.9312, 0.9215, 0.9253, 0.8622, 0.8610, 0.9335, 0.9235, 0.9518,
             0.9440, 0.8250, 0.9058, 0.8623, 0.9463]

val_acc = [0.5430, 0.6192, 0.7096, 0.6918, 0.7277, 0.6838, 0.7690, 0.6401, 0.6901, 0.6465,
           0.6525, 0.6486, 0.7936, 0.7407, 0.7779, 0.7110, 0.6940, 0.7693, 0.8056, 0.8462,
           0.7343, 0.6085, 0.8224, 0.6950, 0.7787]

train_loss = [1.1341, 0.8536, 0.7844, 0.7242, 0.6899, 0.5467, 0.5830, 0.4549, 0.5223, 0.4035,
              0.4744, 0.5662, 0.6550, 0.7549, 0.6475, 0.6901, 0.5939, 0.8496, 0.4963, 0.6242,
              0.6809, 1.1507, 0.7938, np.nan, 1.1612]

val_loss = [0.7556, 0.8056, 0.7933, 0.8985, 0.8817, 1.2538, 1.0651, 2.0207, 2.0194, 2.4561,
            2.2334, 2.9383, 2.8953, 2.9583, 3.2818, 4.5394, 5.1701, 3.3018, np.nan, np.nan,
            np.nan, np.nan, 3.4537, np.nan, np.nan]

# Kreiraj grafik
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Grafikon 1: Accuracy ---
ax1.plot(epochs, [a * 100 for a in train_acc], 'b-o', label='Trening tačnost', markersize=5, linewidth=1.5)
ax1.plot(epochs, [a * 100 for a in val_acc], 'r-o', label='Validaciona tačnost', markersize=5, linewidth=1.5)

# Označi ključne tačke
ax1.annotate(f'Najbolja: {84.6}%', xy=(20, 84.6), xytext=(15, 88),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green', fontweight='bold')
ax1.annotate(f'Overfitting\n(razlika ~21%)', xy=(13, 79.4), xytext=(8, 85),
             arrowprops=dict(arrowstyle='->', color='orange'), fontsize=8, color='orange')

ax1.set_xlabel('Epoha', fontsize=11)
ax1.set_ylabel('Tačnost (%)', fontsize=11)
ax1.set_title('Trening i Validaciona Tačnost', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(40, 100)

# --- Grafikon 2: Loss ---
ax2.plot(epochs, train_loss, 'b-o', label='Trening gubitak (Loss)', markersize=5, linewidth=1.5)
ax2.plot(epochs, val_loss, 'r-o', label='Validacioni gubitak (Loss)', markersize=5, linewidth=1.5)

# Označi ključne tačke
ax2.annotate('Eksplodirajući\ngradijent\n(Infinity)', xy=(19, 5.5), xytext=(21, 7),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red', fontweight='bold')
ax2.annotate(f'Najmanji: {0.40:.2f}', xy=(10, 0.40), xytext=(6, 2.0),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green', fontweight='bold')

ax2.set_xlabel('Epoha', fontsize=11)
ax2.set_ylabel('Gubitak (Loss)', fontsize=11)
ax2.set_title('Trening i Validacioni Gubitak (Loss)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150)
plt.show()
print("Grafik sačuvan u results/training_curves.png")