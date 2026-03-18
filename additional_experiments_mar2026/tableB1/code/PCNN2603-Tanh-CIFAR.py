import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import tarfile
import pickle
import csv

# ====================== 纯CPU环境配置 ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 确保data目录存在
if not os.path.exists('data'):
    os.makedirs('data')

# ====================== 工具函数：保存模型参数（兼容批归一化层） ======================
def save_model_weights_biases(model, filepath):
    """保存模型weights和biases到指定路径，兼容所有层类型"""
    params = {}
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:  # 仅保存有参数的层
            # 不同层的参数处理：兼容2个/4个参数的情况
            if len(layer_weights) == 2:  # 卷积/全连接层（weights + biases）
                weights, biases = layer_weights
                params[layer.name] = {
                    'weights': weights,
                    'biases': biases
                }
            elif len(layer_weights) == 4:  # 批归一化层（gamma, beta, moving_mean, moving_variance）
                gamma, beta, moving_mean, moving_variance = layer_weights
                params[layer.name] = {
                    'gamma': gamma,
                    'beta': beta,
                    'moving_mean': moving_mean,
                    'moving_variance': moving_variance
                }
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"Biases and Weights saved to: {filepath}")

# ====================== 1. 加载本地CIFAR-10数据集 ======================
data_dir = os.path.join(os.path.dirname(__file__), 'data')
local_tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')

if not os.path.exists(local_tar_path):
    raise FileNotFoundError(f"未找到数据集文件，请确认路径：{local_tar_path}")

def load_cifar10_from_local(tar_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    cifar10_dir = os.path.join(data_dir, 'cifar-10-batches-py')

    x_train = []
    y_train = []
    for i in range(1, 6):
        batch_file = os.path.join(cifar10_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            x_train.append(batch['data'])
            y_train.append(batch['labels'])
    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.concatenate(y_train)

    test_file = os.path.join(cifar10_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        x_test = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_test = np.array(batch['labels'])  # 保留原始标签（数字）

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cifar10_from_local(local_tar_path)

# ====================== 2. 数据预处理 ======================
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 独热编码仅用于训练/评估，统计时用原始标签
y_train_onehot = utils.to_categorical(y_train, 10)
y_test_onehot = utils.to_categorical(y_test, 10)

# ====================== 3. 构建双子网CNN模型（增加可训练参数） ======================
input_layer = layers.Input(shape=(32, 32, 3), name='input_layer')

# 子网1的层（大幅增加参数）
conv1_sub1 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same', name='conv1_sub1')(input_layer)  # 16→32
conv1_sub1 = layers.BatchNormalization()(conv1_sub1)
pool1_sub1 = layers.MaxPooling2D((2, 2), name='pool1_sub1')(conv1_sub1)

conv2_sub1 = layers.Conv2D(64, (3, 3), activation='tanh', padding='same', name='conv2_sub1')(pool1_sub1)  # 32→64
conv2_sub1 = layers.BatchNormalization()(conv2_sub1)
pool2_sub1 = layers.MaxPooling2D((2, 2), name='pool2_sub1')(conv2_sub1)

# 新增第三层卷积（额外增加参数）
conv3_sub1 = layers.Conv2D(128, (3, 3), activation='tanh', padding='same', name='conv3_sub1')(pool2_sub1)
conv3_sub1 = layers.BatchNormalization()(conv3_sub1)
pool3_sub1 = layers.MaxPooling2D((2, 2), name='pool3_sub1')(conv3_sub1)

flatten_sub1 = layers.Flatten(name='flatten_sub1')(pool3_sub1)
dense1_sub1 = layers.Dense(128, activation='tanh', name='dense1_sub1')(flatten_sub1)  # 64→128
dropout1_sub1 = layers.Dropout(0.3, name='dropout1_sub1')(dense1_sub1)
sub1_output_layer = layers.Dense(10, activation='softmax', name='sub1_output')(dropout1_sub1)

# 子网2的层（和子网1完全一致，保证结构对称）
conv1_sub2 = layers.Conv2D(32, (3, 3), activation='tanh', padding='same', name='conv1_sub2')(input_layer)  # 16→32
conv1_sub2 = layers.BatchNormalization()(conv1_sub2)
pool1_sub2 = layers.MaxPooling2D((2, 2), name='pool1_sub2')(conv1_sub2)

conv2_sub2 = layers.Conv2D(64, (3, 3), activation='tanh', padding='same', name='conv2_sub2')(pool1_sub2)  # 32→64
conv2_sub2 = layers.BatchNormalization()(conv2_sub2)
pool2_sub2 = layers.MaxPooling2D((2, 2), name='pool2_sub2')(conv2_sub2)

# 新增第三层卷积
conv3_sub2 = layers.Conv2D(128, (3, 3), activation='tanh', padding='same', name='conv3_sub2')(pool2_sub2)
conv3_sub2 = layers.BatchNormalization()(conv3_sub2)
pool3_sub2 = layers.MaxPooling2D((2, 2), name='pool3_sub2')(conv3_sub2)

flatten_sub2 = layers.Flatten(name='flatten_sub2')(pool3_sub2)
dense1_sub2 = layers.Dense(128, activation='tanh', name='dense1_sub2')(flatten_sub2)  # 64→128
dropout1_sub2 = layers.Dropout(0.3, name='dropout1_sub2')(dense1_sub2)
sub2_output_layer = layers.Dense(10, activation='softmax', name='sub2_output')(dropout1_sub2)

# 融合两个子网的特征（全连接层输出）
merged_features = layers.Average(name='merged_average')([dropout1_sub1, dropout1_sub2])
whole_net_output = layers.Dense(10, activation='softmax', name='whole_net_output')(merged_features)

# 构建模型
model = models.Model(inputs=input_layer, outputs=whole_net_output)
subnet1_model = models.Model(inputs=input_layer, outputs=sub1_output_layer)
subnet2_model = models.Model(inputs=input_layer, outputs=sub2_output_layer)

# ====================== 4. 配置优化器+学习率（0.002） ======================
adam_optimizer = Adam(learning_rate=0.002)
model.compile(
    optimizer=adam_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# 打印模型参数总量（验证参数增加效果）
total_params = model.count_params()
print(f"Number of parameters: {total_params:,}")  # 原模型约18万，新模型约85万
model.summary()

# ====================== 5. 定义回调函数（学习率衰减+早停） ======================
# 学习率衰减：验证集准确率5轮不提升，学习率减半
lr_reduce = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# 早停：验证集准确率10轮不提升则停止，保留最优模型
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

callbacks_list = [lr_reduce, early_stop]

# ====================== 6. 保存参数初始值 ======================
save_model_weights_biases(model, 'data/biases&weights_initialized.pickle')

# ====================== 7. 模型训练（带回调） ======================
history = model.fit(
    x_train, y_train_onehot,
    batch_size=128,
    epochs=50,
    validation_split=0.1,
    shuffle=True,
    verbose=2,
    callbacks=callbacks_list
)

# ====================== 8. 保存参数训练后值 ======================
save_model_weights_biases(model, 'data/biases&weights_optimized.pickle')

# ====================== 9. 保存训练/验证准确率到CSV ======================
accuracy_data = [['epoch', 'train_accuracy', 'val_accuracy']]
for epoch in range(len(history.history['accuracy'])):
    accuracy_data.append([
        epoch+1,
        round(history.history['accuracy'][epoch], 4),
        round(history.history['val_accuracy'][epoch], 4)
    ])

with open('data/classification_accuracy.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(accuracy_data)

# ====================== 10. 模型评估 ======================
num_of_CPU_processors = os.cpu_count()
test_loss, test_acc = model.evaluate(
    x_test, y_test_onehot,
    batch_size=10000,
    use_multiprocessing=True,
    workers=num_of_CPU_processors
)
print(f"\nLoss on evaluation data: {test_loss:.4f}")
print(f"Accuracy on evaluation data: {test_acc:.4f}")

# ====================== 11. 追加测试结果到CSV ======================
with open('data/classification_accuracy.csv', 'r', encoding='utf-8') as f:
    existing_data = list(csv.reader(f))

existing_data.append([])
existing_data.append(['test_loss', 'test_accuracy'])
existing_data.append([round(test_loss, 4), round(test_acc, 4)])

with open('data/classification_accuracy.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(existing_data)

# ====================== 12. 统计子网输出差异 ======================
print("\nCounting the numbers of different types of results ...")
whole_net_output_pred = model.predict(x_test, batch_size=10000, use_multiprocessing=True, workers=num_of_CPU_processors)
subnet1_output_pred = subnet1_model.predict(x_test, batch_size=10000, use_multiprocessing=True, workers=num_of_CPU_processors)
subnet2_output_pred = subnet2_model.predict(x_test, batch_size=10000, use_multiprocessing=True, workers=num_of_CPU_processors)

# 计算分类结果
rp = np.argmax(whole_net_output_pred, axis=1)
r1 = np.argmax(subnet1_output_pred, axis=1)
r2 = np.argmax(subnet2_output_pred, axis=1)
y = y_test

# 统计各类结果
type1_count = 0  # rp=r1=r2=y
type2_count = 0  # rp=r2=y, r1!=y
type3_count = 0  # rp=r1=y, r2!=y
type4_count = 0  # rp=y, r1!=y, r2!=y
type4_details = []

for idx in range(len(x_test)):
    current_rp = rp[idx]
    current_r1 = r1[idx]
    current_r2 = r2[idx]
    current_y = y[idx]
    
    if current_rp == current_y and current_r1 == current_y and current_r2 == current_y:
        type1_count += 1
    elif current_rp == current_y and current_r2 == current_y and current_r1 != current_y:
        type2_count += 1
    elif current_rp == current_y and current_r1 == current_y and current_r2 != current_y:
        type3_count += 1
    elif current_rp == current_y and current_r1 != current_y and current_r2 != current_y:
        type4_count += 1
        type4_details.append([idx, int(current_rp), int(current_r1), int(current_r2)])

# 保存统计结果到CSV
csv_data = []
csv_data.append(['Accuracy on the validation data', (type1_count + type2_count + type3_count + type4_count) / len(x_test)])
csv_data.append([])
csv_data.append(['Number of Type I results', type1_count])
csv_data.append(['Number of Type II results', type2_count])
csv_data.append(['Number of Type III results', type3_count])
csv_data.append(['Number of Type IV results', type4_count])
csv_data.append([])
csv_data.append(['Ratio of Type IV results to correct results', type4_count / (type1_count + type2_count + type3_count + type4_count)])
csv_data.append([])
csv_data.append(['Details of Type IV results'])
csv_data.append(['Input index', 'rp', 'r1', 'r2'])
csv_data.extend(type4_details)

with open('data/disagreed_results.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

# 打印最终统计结果
print("\n===== Result counts =====")
print(f"Type I (rp=r1=r2=y): {type1_count}")
print(f"Type II (rp=r2=y, r1!=y): {type2_count}")
print(f"Type III (rp=r1=y, r2!=y): {type3_count}")
print(f"Type IV (rp=y, r1!=y, r2!=y): {type4_count}")
print(f"\nAll results saved to: data/disagreed_results.csv")
