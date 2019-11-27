# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import sys
import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

# 第一步: 在下面这个地址下载语料库
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """
    这个函数的功能是：
    如果filename不存在，就在上面的地址下载它。 如果filename存在，就跳过下载。 最终会检查文字的字节数是否和expected_bytes相同。
    """
    if not os.path.exists(filename):
        print('start downloading...')
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    # return filename


# 下载语料库text8.zip并验证下载
filename = "text8.zip"
maybe_download(filename, 31344016)


# 将语料库解压，并转换成一个word的list
def read_data(filename):
    """
    将下载好的zip文件解压并读取为word的list
    """
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
# print(type(vocabulary))  # list
# print(len(vocabulary), vocabulary[0:3]) # 17005207 ['anarchism', 'originated', 'as', 'a', 'term']

# 第二步: 制作一个词表，将不常见的词变成一个UNK标识符
# 词表的大小为5万（即我们只考虑最常出现的5万个词）
def build_dataset(words, n_words):
    """
    函数功能：将原始的单词表示变成index
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # UNK的index为0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


vocabulary_size = 5000
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

# print(len(data), data[:10])  # 所有字, 在字典中的值
# # 17005207 [0, 0, 12, 6, 195, 2, 0, 46, 59, 156]
#
# print(count)          # 所有字按照出现的多少排序,第一个是所有非常用字的总和
# # [['UNK', 6747638], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)
#
# print(dictionary)           # 字-->排序
# # {'UNK': 0, 'the': 1, 'of': 2, 'and': 3, 'one': 4, 'in': 5,
#
# print(reverse_dictionary)  # 排序-->字
# # {0: 'UNK', 1: 'the', 2: 'of', 3: 'and', 4: 'one', 5: 'in'


del vocabulary  # 删除已节省内存
# # 输出最常出现的5个单词
# print('最常出现的5个单词', count[:5])
# # 输出转换后的数据库 data，和原来的单词（前10个）
print(data[:8], [reverse_dictionary[i] for i in data[:8]])


# 第三步：定义一个函数，用于生成 skip-gram 模型用的 batch
# 我们下面就使用data来制作训练集
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    # data_index相当于一个指针，初始为0
    # 每次生成一个batch，data_index就会相应地往后推
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # 3
    buffer = collections.deque(maxlen=span)  # 创建一个装 3 个数的队列
    # data_index是当前数据开始的位置 产生batch后就往后推1位（产生batch）
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # print("buffer:", buffer)  # deque([0, 0, 12], maxlen=3)
    # print("data_index:", data_index)  # 3

    for i in range(batch_size // num_skips):  # [0, 1, 2, 3] # i = 0
        target = skip_window  # 1
        targets_to_avoid = [skip_window]  # targets_to_avoid = [1, 0]

        for j in range(num_skips):  # [0, 1]  # j = 0
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)  # [0, 1, 2] # 0
            targets_to_avoid.append(target)
            # print("  ", buffer[skip_window])
            batch[i * num_skips + j] = buffer[skip_window]  # batch[0]=buffer[1] # batch[1]=buffer[1]
            labels[i * num_skips + j, 0] = buffer[target]   # labels[0, 0]=buffer[0] # labels[1, 0]=buffer[2]

        buffer.append(data[data_index])
        # print(i, "buffer append 后:", buffer)
        data_index = (data_index + 1) % len(data)  # 每利用buffer生成num_skips个样本，data_index就向后推进一位
    # print("??11", data_index, data[data_index])
    data_index = (data_index + len(data) - span) % len(data)
    # print("??22", data_index, data[data_index])
    return batch, labels


# 默认情况下skip_window=1, num_skips=2
# 此时就是从连续的3(3 = skip_window*2 + 1)个词中生成2(num_skips)个样本。
# 如连续的三个词['used', 'against', 'early']
# 生成两个样本：against -> used, against -> early
batch_size_test = 8
batch, labels = generate_batch(batch_size=batch_size_test, num_skips=2, skip_window=2)

for i in range(batch_size_test):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
# 0 UNK -> 12 as
# 0 UNK -> 0 UNK
# 12 as -> 6 a
# 12 as -> 0 UNK
# 6 a -> 195 term
# 6 a -> 12 as
# 195 term -> 6 a
# 195 term -> 2 of
# exit()

# 第四步: 建立模型
# valid_size = 5  # 每次验证 5 个词
# valid_window = 100  # 这 5 个词是在前100个最常见的词中选出来的
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# print("验证的词的索引:", valid_examples, type(valid_examples))
valid_examples = np.array([4, 7, 25, 32, 15])


batch_size = 64
embedding_size = 128  # 词嵌入空间是 128 维
skip_window = 1  # skip_window 参数和之前保持一致
num_skips = 2  # num_skips 参数和之前保持一致
graph = tf.Graph()
with graph.as_default():
    # 输入的batch
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    print("train_inputs :", train_inputs)  # Tensor("Placeholder:0", shape=(64,), dtype=int32)

    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    print(" train_labels:", train_labels)  # Tensor("Placeholder_1:0", shape=(64, 1), dtype=int32)

    # 用于验证的词
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    print("valid_dataset :", valid_dataset)  # Tensor("Const:0", shape=(5,), dtype=int32)

    # 定义变量
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    print("embeddings :", embeddings)  # <tf.Variable 'Variable:0' shape=(5000, 128) dtype=float32_ref>

    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    print("embed :", embed)  # Tensor("embedding_lookup/Identity:0", shape=(64, 128), dtype=float32)

    # 创建两个变量用于NCE Loss（即选取噪声词的二分类损失）
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    print(" nce_weights:", nce_weights)  # <tf.Variable 'Variable_1:0' shape=(5000, 128) dtype=float32_ref>

    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    print("nce_biases :", nce_biases)  # <tf.Variable 'Variable_2:0' shape=(5000,) dtype=float32_ref>

    # tf.nn.nce_loss会自动选取噪声词，并且形成损失。
    # 随机选取num_sampled个噪声词
    num_sampled = 32
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                                         num_sampled=num_sampled, num_classes=vocabulary_size))
    print(" loss:", loss)  # Tensor("Mean:0", shape=(), dtype=float32)
    # 构造优化器了
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # 计算词和词的相似度（用于验证）
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
    print("norm :", norm)  # Tensor("Sqrt:0", shape=(5000, 1), dtype=float32)

    normalized_embeddings = embeddings / norm
    print("normalized_embeddings :", normalized_embeddings)  # Tensor("truediv:0", shape=(5000, 128), dtype=float32)

    # 找出和验证词的embedding并计算它们和所有单词的相似度
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    print("valid_embeddings :", valid_embeddings)
    # Tensor("embedding_lookup_1/Identity:0", shape=(5, 128), dtype=float32)

    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    print("similarity :", similarity)  # Tensor("MatMul_1:0", shape=(5, 5000), dtype=float32)

    # 变量初始化步骤
    init0 = tf.global_variables_initializer()

# 第五步：开始训练
num_steps = 50001
with tf.Session(graph=graph) as session:
    init0.run()
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        if step % 20000 == 0:
            if step > 0:
                average_loss /= 20000
            print(step, ' 20000个batch的平均每个batch损失: ', average_loss)
            average_loss = 0

        # 每 1 万步，我们进行一次验证
        if step % 10000 == 0:
            # sim是验证词与所有词之间的相似度
            sim = similarity.eval()
            # 查看 验证词 效果
            for i in xrange(len(valid_examples)):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 5  # 输出最相邻的 5 个词语
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = '\n\"%s\"最近的词是:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s ' % (log_str, close_word)
                print(log_str)
    # final_embeddings 是我们最后得到的 embedding 向量
    # 它的形状是[vocabulary_size, embedding_size]
    # 每一行就代表着对应index词的词嵌入表示
    # print("损失:", nce_biases.eval())
    # print("类型:", type(embeddings.eval()))
    # print("embeddings:", embeddings.eval().shape, embeddings)
    final_embeddings = normalized_embeddings.eval()
    # print("final_embeddings:", final_embeddings)
    # print("valid_dataset具体数据:", valid_dataset.eval())  # [ 5  1 95 22 62]


# Step 6: 可视化
# 因为我们的 embedding 的大小为128维，没有办法直接可视化 所以用t-SNE方法进行降维
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# 只画出500个词的位置
plot_only = 500
print("降维前形状:", final_embeddings[:plot_only, :].shape)  # 500 128
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
print("降维后形状:", low_dim_embs.shape)  # 500 2
labels = [reverse_dictionary[i] for i in xrange(plot_only)]


def plot_with_labels(low_dim_embs, labels, filename='500.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


plot_with_labels(low_dim_embs, labels)
