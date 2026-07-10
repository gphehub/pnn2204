# ====================== 强制配置matplotlib后端（解决无显示问题） ======================
import matplotlib
matplotlib.use('TkAgg')  # Windows用TkAgg，macOS替换为'MacOSX'，Linux用'QtAgg'
# ====================== 基础库导入 ======================
import csv
import pickle
import json
import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# ====================== 激活函数定义 ======================
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0 + np.exp(-z))
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
# 全局激活函数（与原代码保持一致）
activation_fn = sigmoid
activation_fn_prime = sigmoid_prime

# ====================== 新增：显著性图IoU计算函数 ======================
def calculate_saliency_iou(map1, map2, threshold=0.5):
    """
    计算两张显著性图的IoU（交并比）
    :param map1: Network1 28×28显著性图
    :param map2: Network2 28×28显著性图
    :param threshold: 二值化阈值，大于阈值视为显著区域
    :return: IoU数值 [0,1]
    """
    mask1 = (map1 > threshold).astype(np.float32)
    mask2 = (map2 > threshold).astype(np.float32)

    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))

    if union < 1e-8:
        return 0.0
    return intersection / union

# ====================== Type分类与样本筛选函数（v6：偶数列y值不同） ======================
def classify_result_type(y, p, a, b):
    """
    根据预测结果分类Type
    :param y: 真实标签
    :param p: 融合网络预测
    :param a: NN1p预测
    :param b: NN2p预测
    :return: Type名称（I/II/III/IV）
    """
    if p == y and a == y and b == y:
        return "Type I"
    elif p == y and b == y and a != y:
        return "Type II"
    elif p == y and a == y and b != y:
        return "Type III"
    elif p == y and a != y and b != y:
        return "Type IV"
    else:
        return "Other"  # 非目标类型
def select_samples_by_type_v6(net, test_data, max_search=50000):
    """
    V6版本：筛选8个样本（2×I/II/III/IV），要求偶数列y值与奇数列不同
    顺序：Type I(奇) → Type I(偶，y不同) → Type II(奇) → Type II(偶，y不同) → Type III(奇) → Type III(偶，y不同) → Type IV(奇) → Type IV(偶，y不同)
    """
    target_types = ["Type I", "Type II", "Type III", "Type IV"]
    selected_samples = {t: {"odd": None, "even": None} for t in target_types}  # 奇/偶列样本
    searched_count = 0
    
    print("Searching for 8 samples (2×Type I/II/III/IV) with different y values for even columns...")
    
    # 遍历样本，筛选目标Type
    for idx, (x_flat_col, y) in enumerate(test_data):
        if searched_count >= max_search:
            break
        if all([selected_samples[t]["odd"] and selected_samples[t]["even"] for t in target_types]):
            break
        
        x_flat_1d = x_flat_col.reshape(-1)
        x_img = x_flat_1d.reshape(28, 28)
        
        p = np.argmax(net.feedforwardPara(x_flat_1d))
        a = np.argmax(net.feedforwardNN1p(x_flat_1d))
        b = np.argmax(net.feedforwardNN2p(x_flat_1d))
        
        r_type = classify_result_type(y, p, a, b)
        if r_type not in target_types:
            continue
        
        if selected_samples[r_type]["odd"] is None:
            selected_samples[r_type]["odd"] = {
                "index": idx,
                "img": x_img,
                "flat": x_flat_1d,
                "label": y,
                "type": r_type,
                "p": p, "a": a, "b": b
            }
            print(f"Found {r_type} (odd column) at index {idx}: y={y}, p={p}, a={a}, b={b}")
        elif selected_samples[r_type]["even"] is None and y != selected_samples[r_type]["odd"]["label"]:
            selected_samples[r_type]["even"] = {
                "index": idx,
                "img": x_img,
                "flat": x_flat_1d,
                "label": y,
                "type": r_type,
                "p": p, "a": a, "b": b
            }
            print(f"Found {r_type} (even column, y≠{selected_samples[r_type]['odd']['label']}) at index {idx}: y={y}, p={p}, a={a}, b={b}")
        
        searched_count += 1
    
    missing = []
    for t in target_types:
        if not selected_samples[t]["odd"]:
            missing.append(f"{t} (odd column)")
        if not selected_samples[t]["even"]:
            missing.append(f"{t} (even column)")
    if missing:
        raise ValueError(f"Failed to collect enough samples: {', '.join(missing)} (searched {searched_count} samples)")
    
    result = []
    for t in target_types:
        result.append(selected_samples[t]["odd"])
        result.append(selected_samples[t]["even"])
    
    return result

# ====================== 显著性图计算 ======================
def compute_saliency_map(net, x_flat, target_label, is_nn1=True):
    """
    计算单个样本的显著性图
    :param net: Network实例
    :param x_flat: 展平的输入样本（784维，一维数组）
    :param target_label: 样本真实标签
    :param is_nn1: True=计算Network1的显著性，False=计算Network2的显著性
    :return: 28×28的显著性图（归一化到[0,1]）
    """
    a = x_flat.copy().reshape(-1, 1)
    zs = []
    activations = [a]
    
    biases = net.biasesNN1 if is_nn1 else net.biasesNN2
    weights = net.weightsNN1 if is_nn1 else net.weightsNN2
    
    for b, w in zip(biases, weights):
        z = np.dot(w, a) + b
        zs.append(z)
        a = activation_fn(z)
        activations.append(a)
    
    delta = activations[-1]
    delta[target_label] -= 1
    
    for l in range(len(zs)-1, 0, -1):
        delta = np.dot(weights[l].T, delta) * activation_fn_prime(zs[l-1])
    
    input_gradient = np.dot(weights[0].T, delta)
    saliency = np.abs(input_gradient).reshape(28, 28)
    if saliency.max() - saliency.min() > 1e-8:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    else:
        saliency = np.zeros_like(saliency)
    return saliency

# ====================== 绘图函数（分段GridSpec，仅压缩Network2与IoU行间距离 + 调整色条高度位置） ======================
def plot_saliency_maps_v6(selected_samples, net):
    """
    V6版本绘图：
    - 第2/4/6/8列y值与第1/3/5/7列不同
    - 第二/三行标题NN改为Network
    - 底部增加IoU行，仅缩小IoU与上方Network2图的间距
    - 右侧色条调整高度与垂直位置，上下不超出左侧绘图区域
    """
    original_imgs = [s["img"] for s in selected_samples]
    nn1_saliency_list = [compute_saliency_map(net, s["flat"], s["label"], is_nn1=True) for s in selected_samples]
    nn2_saliency_list = [compute_saliency_map(net, s["flat"], s["label"], is_nn1=False) for s in selected_samples]
    indices = [s["index"] for s in selected_samples]
    labels = [s["label"] for s in selected_samples]
    a_values = [s["a"] for s in selected_samples]
    b_values = [s["b"] for s in selected_samples]
    result_types = [s["type"] for s in selected_samples]

    iou_list = []
    for m1, m2 in zip(nn1_saliency_list, nn2_saliency_list):
        iou = calculate_saliency_iou(m1, m2, threshold=0.5)
        iou_list.append(iou)

    # ========== 分段网格布局：前3行宽松，第3/4行极小间距 ==========
    fig = plt.figure(figsize=(20, 11))
    # 上3行：原图、Network1、Network2，正常行间距
    gs_upper = fig.add_gridspec(3, 8, hspace=0.15, left=0, right=0.92, top=0.96, bottom=0.33)
    # 最下方IoU单行，紧贴上面，极小间距
    gs_iou = fig.add_gridspec(1, 8, hspace=0.01, left=0, right=0.92, top=0.29, bottom=0.02)

    axes = np.empty((4, 8), dtype=object)
    # 填充前3行子图
    for r in range(3):
        for c in range(8):
            axes[r, c] = fig.add_subplot(gs_upper[r, c])
    # 填充IoU第4行
    for c in range(8):
        axes[3, c] = fig.add_subplot(gs_iou[0, c])

    # 第1行：原图
    for i, (img, idx, label) in enumerate(zip(original_imgs, indices, labels)):
        ax = axes[0, i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Input image index={idx}\nlabel={label}", fontsize=9)
        ax.axis('off')

    # 第2行：Network1显著性图
    for i, (saliency, a_val) in enumerate(zip(nn1_saliency_list, a_values)):
        ax = axes[1, i]
        im = ax.imshow(saliency, cmap='hot')
        ax.set_title(f"Network1 (output={a_val})", fontsize=9)
        ax.axis('off')

    # 第3行：Network2显著性图
    for i, (saliency, b_val, r_type) in enumerate(zip(nn2_saliency_list, b_values, result_types)):
        ax = axes[2, i]
        im = ax.imshow(saliency, cmap='hot')
        ax.set_title(f"Network2 (output={b_val})", fontsize=9)
        ax.axis('off')
        ax.text(0.5, -0.15, r_type, transform=ax.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold', color='red')

    # 第4行IoU文字，y=0.85向上贴近上方图表
    for i, iou_val in enumerate(iou_list):
        ax = axes[3, i]
        ax.text(0.5, 0.85, f"IoU={iou_val:.3f}", transform=ax.transAxes,
                ha='center', va='center', fontsize=10, fontweight='bold', color='darkblue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # ==================== 修改右侧色条坐标，上下不超出左侧绘图区域 ====================
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.93, 0.24, 0.015, 0.50])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Saliency Value (Normalized)', fontsize=10)

    # 取消tight_layout，手动网格已控制位置
    # plt.tight_layout(rect=[0, 0.02, 0.92, 0.96], h_pad=0.3)
    plt.savefig('he_fig9.png', dpi=300, bbox_inches='tight')
    print("Saliency map (v6 IoU compact spacing + adjusted colorbar) saved as 'he_fig9.png'")
    print("Each column IoU ratio (Network1 vs Network2 saliency):")
    for col_idx, iou in enumerate(iou_list):
        print(f"Column {col_idx+1}: IoU = {iou:.3f}")
    plt.show()

# ====================== 外部调用可视化接口 ======================
def generate_mnist_saliency_maps_v6(net, test_data):
    selected_samples = select_samples_by_type_v6(net, test_data)
    print("\n===== V6 Selected Samples Summary =====")
    col_types = ["1 (odd)", "2 (even)", "3 (odd)", "4 (even)", "5 (odd)", "6 (even)", "7 (odd)", "8 (even)"]
    for i, s in enumerate(selected_samples):
        print(f"Column {col_types[i]} - {s['type']}: index={s['index']}, label={s['label']}, p={s['p']}, a={s['a']}, b={s['b']}")
    plot_saliency_maps_v6(selected_samples, net)

# ====================== 双子网核心类 ======================
class Network(object):
    def __init__(self):
        try:
            self.saved_weight_loader('biases&weights_optimized.pickle')
            print("Loaded pretrained weights from 'biases&weights_optimized.pickle'")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No valid pretrained weights found, generating random weights for MNIST (784→128→64→10)")
            self._init_mnist_weights()
    def _init_mnist_weights(self):
        self.sizesNN1 = [784, 128, 64, 10]
        self.biasesNN1 = [np.random.randn(y, 1) for y in self.sizesNN1[1:]]
        self.weightsNN1 = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizesNN1[:-1], self.sizesNN1[1:])]
        self.sizesNN2 = [784, 128, 64, 10]
        self.biasesNN2 = [np.random.randn(y, 1) for y in self.sizesNN2[1:]]
        self.weightsNN2 = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizesNN2[:-1], self.sizesNN2[1:])]
    def saved_weight_loader(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        self.sizesNN1 = data["sizes1"]
        self.biasesNN1 = [np.array(b) for b in data["biases1"]]
        self.weightsNN1 = [np.array(w) for w in data["weights1"]]
        self.sizesNN2 = data["sizes2"]
        self.biasesNN2 = [np.array(b) for b in data["biases2"]]
        self.weightsNN2 = [np.array(w) for w in data["weights2"]]
    def separated_weight_loader(self):
        with open('biases&weights_network1.pickle', "r") as f:
            data = json.load(f)
        self.sizesNN1 = data["sizes"]
        self.biasesNN1 = [np.array(b) for b in data["biases"]]
        self.weightsNN1 = [np.array(w) for w in data["weights"]]
        with open('biases&weights_network2.pickle', "r") as f:
            data = json.load(f)
        self.sizesNN2 = data["sizes"]
        self.biasesNN2 = [np.array(b) for b in data["biases"]]
        self.weightsNN2 = [np.array(w) for w in data["weights"]]
    def feedforwardNN1(self, a):
        a = a.reshape(-1, 1)
        for b, w in zip(self.biasesNN1, self.weightsNN1):
            a = activation_fn(np.dot(w, a) + b)
        return a.flatten()
    def feedforwardNN2(self, a):
        a = a.reshape(-1, 1)
        for b, w in zip(self.biasesNN2, self.weightsNN2):
            a = activation_fn(np.dot(w, a) + b)
        return a.flatten()
    def feedforwardNN1p(self, a):
        a = a.reshape(-1, 1)
        for b, w in zip(self.biasesNN1, self.weightsNN1):
            z = np.dot(w, a) + b
            a = activation_fn(z)
        a = activation_fn(z + self.biasesNN2[-1])
        return a.flatten()
    def feedforwardNN2p(self, a):
        a = a.reshape(-1, 1)
        for b, w in zip(self.biasesNN2, self.weightsNN2):
            z = np.dot(w, a) + b
            a = activation_fn(z)
        a = activation_fn(z + self.biasesNN1[-1])
        return a.flatten()
    def feedforwardPara(self, a):
        a1 = a.reshape(-1, 1)
        a2 = a.reshape(-1, 1)
        for b, w in zip(self.biasesNN1, self.weightsNN1):
            z1 = np.dot(w, a1) + b
            a1 = activation_fn(z1)
        for b, w in zip(self.biasesNN2, self.weightsNN2):
            z2 = np.dot(w, a2) + b
            a2 = activation_fn(z2)
        a = activation_fn(z1 + z2)
        return a.flatten()
    def SGDmod(self, evaluation_data=None):
        if evaluation_data is None or len(evaluation_data) == 0:
            print("Warning: No evaluation data provided!")
            return
        n_data = len(evaluation_data)
        rateNN1 = self.accuracyNN1(evaluation_data) / n_data
        rateNN2 = self.accuracyNN2(evaluation_data) / n_data
        rateNN1p = self.accuracyNN1p(evaluation_data) / n_data
        rateNN2p = self.accuracyNN2p(evaluation_data) / n_data
        ratePara = self.accuracyPara(evaluation_data) / n_data
        print("\n===== Accuracy Results =====")
        print(f"Accuracy: Network1={rateNN1:.4f}, Network1p={rateNN1p:.4f}, Network2={rateNN2:.4f}, Network2p={rateNN2p:.4f}, Para={ratePara:.4f}")
        print(f"Number of evaluation data: {n_data}")
        with open('disagreed_results.csv', 'w', newline='', encoding='utf-8') as resultsave:
            writer = csv.writer(resultsave)
            writer.writerow(("Network1:", rateNN1))
            writer.writerow(("Network1p:", rateNN1p))
            writer.writerow(("Network2:", rateNN2))
            writer.writerow(("Network2p:", rateNN2p))
            writer.writerow(("Para:", ratePara))
            writer.writerow((""))
            writer.writerow(("Number of evaluation data: ", n_data))
        w1, w2, wb, rb = self.disagreed_results(evaluation_data)
        return wb
    def disagreed_results(self, data, convert=False):
        if convert:
            all_results = [(x, np.argmax(y), np.argmax(self.feedforwardPara(x)),
                          np.argmax(self.feedforwardNN1p(x)),
                          np.argmax(self.feedforwardNN2p(x)))
                       for (x, y) in data]
        else:
            all_results = [(x, y, np.argmax(self.feedforwardPara(x)),
                          np.argmax(self.feedforwardNN1p(x)),
                          np.argmax(self.feedforwardNN2p(x)))
                       for (x, y) in data]
        wrong_1 = sum(int(p == y and a != y and b == y) for (x, y, p, a, b) in all_results)
        wrong_2 = sum(int(p == y and a == y and b != y) for (x, y, p, a, b) in all_results)
        wrong_both = sum(int(p == y and a != y and b != y) for (x, y, p, a, b) in all_results)
        right_both = sum(int(p == y and a == y and b == y) for (x, y, p, a, b) in all_results)
        total_correct = wrong_1 + wrong_2 + wrong_both + right_both
        print("\n===== Subnetwork Disagreement Statistics =====")
        print(f"Total number of correct results: {total_correct}")
        print(f"Number of results when both networks are right: {right_both}")
        print(f"Number of results when only Network1 is wrong: {wrong_1}")
        print(f"Number of results when only Network2 is wrong: {wrong_2}")
        print(f"Number of results when both networks are wrong but combined is right: {wrong_both}")
        with open('disagreed_results.csv', 'a', newline='', encoding='utf-8') as d_resultsave:
            writer = csv.writer(d_resultsave)
            writer.writerow(("Total number of correct results:", total_correct))
            writer.writerow(("Number of results when both networks are right:", right_both))
            writer.writerow(("Number of results when only Network1 is wrong:", wrong_1))
            writer.writerow(("Number of results when only Network2 is wrong:", wrong_2))
            writer.writerow(("Number of results when both networks are wrong but combined is right:", wrong_both))
            writer.writerow((""))
            writer.writerow(("Detailed results when both networks are wrong:", ""))
            writer.writerow(("Combined network", "Network1", "Network2"))
            for (x, y, p, a, b) in all_results:
                if p == y and a != y and b != y:
                    writer.writerow((p, a, b))
        return wrong_1, wrong_2, wrong_both, right_both
    def accuracyNN1(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforwardNN1(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardNN1(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    def accuracyNN2(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforwardNN2(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardNN2(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    def accuracyNN1p(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforwardNN1p(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardNN1p(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    def accuracyNN2p(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforwardNN2p(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardNN2p(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    def accuracyPara(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforwardPara(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardPara(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

# ====================== 测试入口 ======================
if __name__ == "__main__":
    net = Network()
    print("\nNetwork class loaded successfully!")
    try:
        import mnist_loader
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        net.SGDmod(evaluation_data=validation_data)
        generate_mnist_saliency_maps_v6(net, validation_data)
    except ImportError:
        print("mnist_loader not found, skip data loading test")
    except Exception as e:
        print(f"Error: {str(e)}")
