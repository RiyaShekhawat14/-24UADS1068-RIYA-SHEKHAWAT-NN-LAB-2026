import matplotlib.pyplot as plt
import numpy as np
import time

### data (NAND gate) ###
data = [
    (0, 0, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

# ### XOR gate data (uncomment to test XOR) ###
# data = [
#     (0, 0, 0),
#     (0, 1, 1),
#     (1, 0, 1),
#     (1, 1, 0)
# ]

### ideal fixed weight and bias (reference line) ###
w_true = [-1, -1]
b_true = 1.5

#### initial learned weights ####
w1 = 0
w2 = 0
b = 0
epoch = 1

#### interactive graph mode ###
plt.ion()

## for drawing same graph again and again ###
fig, ax = plt.subplots()

#### graph drawing function ####
def draw_graph(epoch, w1, w2, b):
    ax.clear()

    # plot data points
    for x1, x2, y in data:
        if y == 1:
            ax.scatter(x1, x2, color="green", s=100)
        else:
            ax.scatter(x1, x2, color="red", s=100)

    x = np.array([-0.5, 1.5])

    # true boundary (fixed)
    y_true = (-w_true[0] * x - b_true) / w_true[1]
    ax.plot(x, y_true, "k--", label="True boundary (wᵀx)")

    # learned boundary
    if w2 != 0:
        y_pred = (-w1 * x - b) / w2
        ax.plot(x, y_pred, "b", label=f"Predicted boundary (Epoch {epoch})")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Perceptron Learning – Epoch {epoch}")
    ax.legend()
    ax.grid(True)

    plt.pause(1.0)

### training loop ###
while True:
    print(f"\n===== Epoch {epoch} =====")
    print("x1  x2  y  z  y_pred  w1  w2  b")

    error = 0

    for x1, x2, y in data:
        z = x1 * w1 + x2 * w2 + b
        y_pred = 1 if z >= 0 else 0

        if y == 1 and y_pred == 0:
            w1 += x1
            w2 += x2
            b  += 1
            error += 1

        elif y == 0 and y_pred == 1:
            w1 -= x1
            w2 -= x2
            b  -= 1
            error += 1

        print(f" {x1}   {x2}   {y}  {z:>3}     {y_pred}     {w1:>2}  {w2:>2}  {b:>2}")

    draw_graph(epoch, w1, w2, b)
    epoch += 1

    if error == 0:
       print("\nTraining converged")

    # ===== Accuracy & Confusion Matrix =====
       y_true = []
       y_pred_list = []

       for x1_, x2_, y_ in data:
         z_ = x1_*w1 + x2_*w2 + b
         yp_ = 1 if z_ >= 0 else 0
         y_true.append(y_)
         y_pred_list.append(yp_)

       correct = 0
       for i in range(len(y_true)):
         if y_true[i] == y_pred_list[i]:
            correct += 1

       accuracy = (correct / len(y_true)) * 100
       print("Final Accuracy =", accuracy, "%")

       tp = tn = fp = fn = 0
       for yt, yp in zip(y_true, y_pred_list):
          if yt == 1 and yp == 1:
            tp += 1
          elif yt == 0 and yp == 0:
            tn += 1
          elif yt == 0 and yp == 1:
            fp += 1
          elif yt == 1 and yp == 0:
            fn += 1

       print("\nConfusion Matrix")
       print("        Predicted 0   Predicted 1")
       print(f"Actual 0      {tn}              {fp}")
       print(f"Actual 1      {fn}              {tp}")

       break

        
        

plt.ioff()
plt.show()



