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
        # 检查是否所有Type都收集够奇+偶样本
        if all([selected_samples[t]["odd"] and selected_samples[t]["even"] for t in target_types]):
            break
        
        # 转换输入格式
        x_flat_1d = x_flat_col.reshape(-1)
        x_img = x_flat_1d.reshape(28, 28)
        
        # 计算各网络预测
        p = np.argmax(net.feedforwardPara(x_flat_1d))  # 融合网络
        a = np.argmax(net.feedforwardNN1p(x_flat_1d))  # NN1p
        b = np.argmax(net.feedforwardNN2p(x_flat_1d))  # NN2p
        
        # 分类Type
        r_type = classify_result_type(y, p, a, b)
        
        if r_type not in target_types:
            continue
        
        # 处理奇数列样本（先收集）
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
        # 处理偶数列样本（要求y与奇数列不同）
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
    
    # 校验是否收集够数量
    missing = []
    for t in target_types:
        if not selected_samples[t]["odd"]:
            missing.append(f"{t} (odd column)")
        if not selected_samples[t]["even"]:
            missing.append(f"{t} (even column)")
    if missing:
        raise ValueError(f"Failed to collect enough samples: {', '.join(missing)} (searched {searched_count} samples)")
    
    # 按顺序合并样本：I奇→I偶→II奇→II偶→III奇→III偶→IV奇→IV偶
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
    # 前向传播，记录每一层的z和a（用于反向求导）
    a = x_flat.copy().reshape(-1, 1)  # 转为列向量(784,1)
    zs = []  # 记录每一层的z = w*a + b
    activations = [a]  # 记录每一层的激活值
    
    # 选择对应的子网权重/偏置
    biases = net.biasesNN1 if is_nn1 else net.biasesNN2
    weights = net.weightsNN1 if is_nn1 else net.weightsNN2
    
    # 前向传播
    for b, w in zip(biases, weights):
        z = np.dot(w, a) + b
        zs.append(z)
        a = activation_fn(z)
        activations.append(a)
    
    # 反向求导：计算输出对输入的梯度（显著性图核心）
    # 1. 输出层梯度（交叉熵损失对z的导数）
    delta = activations[-1]
    delta[target_label] -= 1  # 交叉熵损失梯度
    
    # 2. 反向传播计算每层梯度
    for l in range(len(zs)-1, 0, -1):
        delta = np.dot(weights[l].T, delta) * activation_fn_prime(zs[l-1])
    
    # 3. 输入层梯度（即显著性值）
    input_gradient = np.dot(weights[0].T, delta)
    
    # 4. 处理梯度：reshape为28×28，取绝对值，归一化
    saliency = np.abs(input_gradient).reshape(28, 28)
    if saliency.max() - saliency.min() > 1e-8:  # 避免除以0
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    else:
        saliency = np.zeros_like(saliency)
    
    return saliency

# ====================== 绘图函数（v6：NN→Network + 偶数列y不同） ======================
def plot_saliency_maps_v6(selected_samples, net):
    """
    V6版本绘图：
    - 第2/4/6/8列y值与第1/3/5/7列不同
    - 第二/三行标题的NN改为Network
    """
    # 提取绘图数据
    original_imgs = [s["img"] for s in selected_samples]
    nn1_saliency_list = [compute_saliency_map(net, s["flat"], s["label"], is_nn1=True) for s in selected_samples]
    nn2_saliency_list = [compute_saliency_map(net, s["flat"], s["label"], is_nn1=False) for s in selected_samples]
    indices = [s["index"] for s in selected_samples]
    labels = [s["label"] for s in selected_samples]
    a_values = [s["a"] for s in selected_samples]
    b_values = [s["b"] for s in selected_samples]
    result_types = [s["type"] for s in selected_samples]
    
    # 设置画布大小（适配8列，调整为宽屏）
    fig, axes = plt.subplots(3, 8, figsize=(20, 9))
    
    # 第1行：原图（灰度，标题：index=xxx + label=yyy）
    for i, (img, idx, label) in enumerate(zip(original_imgs, indices, labels)):
        ax = axes[0, i]
        ax.imshow(img, cmap='gray')
        # 设置两行标题：index=xxx 换行 label=yyy
        ax.set_title(f"Input image index={idx}\nlabel={label}", fontsize=9)
        ax.axis('off')
    
    # 第2行：Network1显著性图（热图，标题：Network1 (output=aaa)）
    for i, (saliency, a_val) in enumerate(zip(nn1_saliency_list, a_values)):
        ax = axes[1, i]
        im = ax.imshow(saliency, cmap='hot')
        ax.set_title(f"Network1 (output={a_val})", fontsize=9)
        ax.axis('off')
    
    # 第3行：Network2显著性图（热图，标题：Network2 (output=bbb) + Type标注）
    for i, (saliency, b_val, r_type) in enumerate(zip(nn2_saliency_list, b_values, result_types)):
        ax = axes[2, i]
        im = ax.imshow(saliency, cmap='hot')
        ax.set_title(f"Network2 (output={b_val})", fontsize=9)
        ax.axis('off')
        # 仅第3行标注Type（加粗红色字体，居中）
        ax.text(0.5, -0.15, r_type, transform=ax.transAxes, 
                ha='center', va='top', fontsize=11, fontweight='bold', color='red')
    
    # 添加统一的颜色条
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Saliency Value (Normalized)', fontsize=10)
    
    # 调整布局（预留Type标注空间）
    plt.tight_layout(rect=[0, 0.08, 0.92, 0.95])
    plt.savefig('he_fig9.png', dpi=300, bbox_inches='tight')
    print("Saliency map (v6) saved as 'he_fig9.png'")
    
    # 显示图像
    plt.show()

# ====================== 外部调用的可视化接口（v6） ======================
def generate_mnist_saliency_maps_v6(net, test_data):
    """
    V6版本可视化接口：筛选8个样本（偶数列y值不同）并绘制v6版显著性图
    """
    # 1. 筛选8个样本（Type I奇→I偶→II奇→II偶→III奇→III偶→IV奇→IV偶，偶数列y不同）
    selected_samples = select_samples_by_type_v6(net, test_data)
    
    # 2. 打印筛选结果汇总
    print("\n===== V6 Selected Samples Summary =====")
    col_types = ["1 (odd)", "2 (even)", "3 (odd)", "4 (even)", "5 (odd)", "6 (even)", "7 (odd)", "8 (even)"]
    for i, s in enumerate(selected_samples):
        print(f"Column {col_types[i]} - {s['type']}: index={s['index']}, label={s['label']}, p={s['p']}, a={s['a']}, b={s['b']}")
    
    # 3. 绘制v6版显著性图
    plot_saliency_maps_v6(selected_samples, net)

# ====================== 双子网核心类 ======================
class Network(object):

    def __init__(self):
        """初始化网络：优先加载预训练权重，无则生成MNIST适配的随机权重"""
        try:
            self.saved_weight_loader('biases&weights_optimized.pickle')
            print("Loaded pretrained weights from 'biases&weights_optimized.pickle'")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No valid pretrained weights found, generating random weights for MNIST (784→128→64→10)")
            self._init_mnist_weights()

    def _init_mnist_weights(self):
        """初始化适配MNIST的随机权重（784→128→64→10）"""
        # 子网1结构
        self.sizesNN1 = [784, 128, 64, 10]
        self.biasesNN1 = [np.random.randn(y, 1) for y in self.sizesNN1[1:]]
        self.weightsNN1 = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizesNN1[:-1], self.sizesNN1[1:])]
        
        # 子网2结构（与子网1相同）
        self.sizesNN2 = [784, 128, 64, 10]
        self.biasesNN2 = [np.random.randn(y, 1) for y in self.sizesNN2[1:]]
        self.weightsNN2 = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizesNN2[:-1], self.sizesNN2[1:])]

    def saved_weight_loader(self, filename):
        """从JSON格式的pickle文件加载预训练权重"""
        with open(filename, "r") as f:
            data = json.load(f)
        
        self.sizesNN1 = data["sizes1"]
        self.biasesNN1 = [np.array(b) for b in data["biases1"]]
        self.weightsNN1 = [np.array(w) for w in data["weights1"]]
        self.sizesNN2 = data["sizes2"]
        self.biasesNN2 = [np.array(b) for b in data["biases2"]]
        self.weightsNN2 = [np.array(w) for w in data["weights2"]]

    def separated_weight_loader(self):
        """加载分离的子网权重"""
        # 加载子网1
        with open('biases&weights_network1.pickle', "r") as f:
            data = json.load(f)
        self.sizesNN1 = data["sizes"]
        self.biasesNN1 = [np.array(b) for b in data["biases"]]
        self.weightsNN1 = [np.array(w) for w in data["weights"]]
        
        # 加载子网2
        with open('biases&weights_network2.pickle', "r") as f:
            data = json.load(f)
        self.sizesNN2 = data["sizes"]
        self.biasesNN2 = [np.array(b) for b in data["biases"]]
        self.weightsNN2 = [np.array(w) for w in data["weights"]]

    def feedforwardNN1(self, a):
        """前向传播：子网1"""
        a = a.reshape(-1, 1)  # 确保是列向量
        for b, w in zip(self.biasesNN1, self.weightsNN1):
            a = activation_fn(np.dot(w, a) + b)
        return a.flatten()

    def feedforwardNN2(self, a):
        """前向传播：子网2"""
        a = a.reshape(-1, 1)  # 确保是列向量
        for b, w in zip(self.biasesNN2, self.weightsNN2):
            a = activation_fn(np.dot(w, a) + b)
        return a.flatten()

    def feedforwardNN1p(self, a):
        """前向传播：子网1变体（共享最后一层偏置）"""
        a = a.reshape(-1, 1)
        for b, w in zip(self.biasesNN1, self.weightsNN1):
            z = np.dot(w, a) + b
            a = activation_fn(z)
        a = activation_fn(z + self.biasesNN2[-1])
        return a.flatten()

    def feedforwardNN2p(self, a):
        """前向传播：子网2变体（共享最后一层偏置）"""
        a = a.reshape(-1, 1)
        for b, w in zip(self.biasesNN2, self.weightsNN2):
            z = np.dot(w, a) + b
            a = activation_fn(z)
        a = activation_fn(z + self.biasesNN1[-1])
        return a.flatten()

    def feedforwardPara(self, a):
        """前向传播：融合网络"""
        a1 = a.reshape(-1, 1)
        a2 = a.reshape(-1, 1)
        
        # 子网1前向
        for b, w in zip(self.biasesNN1, self.weightsNN1):
            z1 = np.dot(w, a1) + b
            a1 = activation_fn(z1)
        
        # 子网2前向
        for b, w in zip(self.biasesNN2, self.weightsNN2):
            z2 = np.dot(w, a2) + b
            a2 = activation_fn(z2)
        
        # 融合输出
        a = activation_fn(z1 + z2)
        return a.flatten()

    def SGDmod(self, evaluation_data=None):
        """评估各子网准确率并保存结果"""
        if evaluation_data is None or len(evaluation_data) == 0:
            print("Warning: No evaluation data provided!")
            return
        
        n_data = len(evaluation_data)
        # 计算各子网准确率
        rateNN1 = self.accuracyNN1(evaluation_data) / n_data
        rateNN2 = self.accuracyNN2(evaluation_data) / n_data
        rateNN1p = self.accuracyNN1p(evaluation_data) / n_data
        rateNN2p = self.accuracyNN2p(evaluation_data) / n_data
        ratePara = self.accuracyPara(evaluation_data) / n_data
        
        # 打印准确率
        print("\n===== Accuracy Results =====")
        print(f"Accuracy: Network1={rateNN1:.4f}, Network1p={rateNN1p:.4f}, Network2={rateNN2:.4f}, Network2p={rateNN2p:.4f}, Para={ratePara:.4f}")
        print(f"Number of evaluation data: {n_data}")

        # 保存准确率到CSV
        with open('disagreed_results.csv', 'w', newline='', encoding='utf-8') as resultsave:
            writer = csv.writer(resultsave)
            writer.writerow(("Network1:", rateNN1))
            writer.writerow(("Network1p:", rateNN1p))
            writer.writerow(("Network2:", rateNN2))
            writer.writerow(("Network2p:", rateNN2p))
            writer.writerow(("Para:", ratePara))
            writer.writerow((""))
            writer.writerow(("Number of evaluation data: ", n_data))

        # 统计子网差异
        w1, w2, wb, rb = self.disagreed_results(evaluation_data)
        return wb

    def disagreed_results(self, data, convert=False):
        """统计子网输出差异并保存详细结果"""
        # 生成结果列表
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
        
        # 统计各类结果
        wrong_1 = sum(int(p == y and a != y and b == y) for (x, y, p, a, b) in all_results)
        wrong_2 = sum(int(p == y and a == y and b != y) for (x, y, p, a, b) in all_results)
        wrong_both = sum(int(p == y and a != y and b != y) for (x, y, p, a, b) in all_results)
        right_both = sum(int(p == y and a == y and b == y) for (x, y, p, a, b) in all_results)
        total_correct = wrong_1 + wrong_2 + wrong_both + right_both

        # 打印统计结果
        print("\n===== Subnetwork Disagreement Statistics =====")
        print(f"Total number of correct results: {total_correct}")
        print(f"Number of results when both networks are right: {right_both}")
        print(f"Number of results when only Network1 is wrong: {wrong_1}")
        print(f"Number of results when only Network2 is wrong: {wrong_2}")
        print(f"Number of results when both networks are wrong but combined is right: {wrong_both}")

        # 保存详细结果到CSV
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
            
            # 保存详细错误案例
            for (x, y, p, a, b) in all_results:
                if p == y and a != y and b != y:
                    writer.writerow((p, a, b))
        
        return wrong_1, wrong_2, wrong_both, right_both

    def accuracyNN1(self, data, convert=False):
        """计算Network1准确率"""
        if convert:
            results = [(np.argmax(self.feedforwardNN1(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardNN1(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def accuracyNN2(self, data, convert=False):
        """计算Network2准确率"""
        if convert:
            results = [(np.argmax(self.feedforwardNN2(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardNN2(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def accuracyNN1p(self, data, convert=False):
        """计算Network1p准确率"""
        if convert:
            results = [(np.argmax(self.feedforwardNN1p(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardNN1p(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def accuracyNN2p(self, data, convert=False):
        """计算Network2p准确率"""
        if convert:
            results = [(np.argmax(self.feedforwardNN2p(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardNN2p(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def accuracyPara(self, data, convert=False):
        """计算融合网络准确率"""
        if convert:
            results = [(np.argmax(self.feedforwardPara(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforwardPara(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

# ====================== 测试代码 ======================
if __name__ == "__main__":
    # 初始化网络
    net = Network()
    print("\nNetwork class loaded successfully!")
    
    # 加载MNIST数据并生成v6版显著性图
    try:
        import mnist_loader
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        
        # 执行准确率评估（可选）
        net.SGDmod(evaluation_data=validation_data)
        
        # 生成v6版显著性图
        generate_mnist_saliency_maps_v6(net, validation_data)
    except ImportError:
        print("mnist_loader not found, skip data loading test")
    except Exception as e:
        print(f"Error: {str(e)}")
        
