"""
Version:
IoU_v1.py:
Q1:
基于这个程序，编写一个可以计算所有输入数据的IoU值的程序，不用输出图形，只显示总输入数据的数量、Type I至IV各自的数量、IoU的平均值（所有数据的平均值及每一种Type的平均值分别输出）



批量计算全部样本IoU统计，无绘图输出
输出内容：
1. 总有效样本数(Type I~IV)
2. Type I / II / III / IV 各自样本数量
3. 全部样本IoU平均值
4. 每一类Type单独IoU平均值
"""
# ====================== 基础库导入 ======================
import csv
import pickle
import json
import random
import sys
import os
import numpy as np

# ====================== 激活函数定义 ======================
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

activation_fn = sigmoid
activation_fn_prime = sigmoid_prime

# ====================== IoU计算函数 ======================
def calculate_saliency_iou(map1, map2, threshold=0.5):
    mask1 = (map1 > threshold).astype(np.float32)
    mask2 = (map2 > threshold).astype(np.float32)
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    if union < 1e-8:
        return 0.0
    return intersection / union

# ====================== Type分类 ======================
def classify_result_type(y, p, a, b):
    if p == y and a == y and b == y:
        return "Type I"
    elif p == y and b == y and a != y:
        return "Type II"
    elif p == y and a == y and b != y:
        return "Type III"
    elif p == y and a != y and b != y:
        return "Type IV"
    else:
        return "Other"

# ====================== 显著性图计算 ======================
def compute_saliency_map(net, x_flat, target_label, is_nn1=True):
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

# ====================== 批量IoU统计主函数 ======================
def calculate_all_saliency_iou_statistics(net, test_data):
    # 存储每类IoU列表
    iou_records = {
        "Type I": [],
        "Type II": [],
        "Type III": [],
        "Type IV": []
    }
    total_sample_count = 0
    print("Start traversing all data to compute IoU...")

    for idx, (x_flat_col, y) in enumerate(test_data):
        x_flat_1d = x_flat_col.reshape(-1)
        # 各子网预测
        p = np.argmax(net.feedforwardPara(x_flat_1d))
        a = np.argmax(net.feedforwardNN1p(x_flat_1d))
        b = np.argmax(net.feedforwardNN2p(x_flat_1d))
        r_type = classify_result_type(y, p, a, b)
        if r_type == "Other":
            continue
        # 计算两张显著性图
        sal1 = compute_saliency_map(net, x_flat_1d, y, is_nn1=True)
        sal2 = compute_saliency_map(net, x_flat_1d, y, is_nn1=False)
        iou_val = calculate_saliency_iou(sal1, sal2, threshold=0.5)
        iou_records[r_type].append(iou_val)
        total_sample_count += 1
        # 每1000条打印进度
        if total_sample_count % 1000 == 0:
            print(f"Processed {total_sample_count} valid samples...")

    print("\n================ IoU Statistics Result ================")
    # 1. 总有效样本数量
    print(f"Total valid samples (Type I~IV): {total_sample_count}")
    # 2. 每类样本数量
    type_count = {}
    all_iou_list = []
    for t in iou_records:
        cnt = len(iou_records[t])
        type_count[t] = cnt
        all_iou_list.extend(iou_records[t])
        print(f"{t} sample number: {cnt}")
    # 3. 全体IoU均值
    if len(all_iou_list) > 0:
        global_mean_iou = np.mean(all_iou_list)
        print(f"\nAverage IoU of all samples: {global_mean_iou:.4f}")
    else:
        print("\nNo valid samples found!")
        return
    # 4. 每一类单独IoU均值
    print("\nAverage IoU for each type:")
    for t in iou_records:
        lst = iou_records[t]
        if len(lst) > 0:
            mean_t = np.mean(lst)
            print(f"{t} average IoU: {mean_t:.4f}")
        else:
            print(f"{t}: No samples")
    return iou_records

# ====================== 双子网网络类（与原程序完全一致） ======================
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

# ====================== 程序入口 ======================
if __name__ == "__main__":
    net = Network()
    print("\nNetwork loaded successfully!")
    try:
        import mnist_loader
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        # 可选：输出准确率统计
        net.SGDmod(evaluation_data=validation_data)
        # 批量计算全部IoU并打印统计
        calculate_all_saliency_iou_statistics(net, validation_data)
    except ImportError:
        print("Error: mnist_loader.py not found in current directory!")
    except Exception as e:
        print(f"Runtime Error: {str(e)}")
