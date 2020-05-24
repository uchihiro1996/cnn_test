import numpy as np

class Softmax:
  #  softmaxの実装　最終的に0-1の範囲で表す機能

  def __init__(self, input_len, nodes):
    # 初期値の分散を減らすためにノード数で割る
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)　#最終的な出力にバイアスをかける（デフォルトのまま実行）

  def forward(self, input):
    '''
    最終的に0-1の確率で表した一次元numpy配列を返す
    inputには任意の入力変数を渡すことができる
    '''
    self.last_input_shape = input.shape　#入力ノード数

    input = input.flatten()　#平坦化させる　一次元にさせる
    self.last_input = input

    input_len, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases  #インプットされた情報を内積して各ノードに重見つけする
    self.last_totals = totals

    exp = np.exp(totals)
    return exp / np.sum(exp, axis=0)

  def backprop(self, d_L_d_out, learn_rate):
    '''
    損失率の算出プロセス及び重みの更新
    '''
    # We know only 1 element of d_L_d_out will be nonzero
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      # e^totals
      t_exp = np.exp(self.last_totals)

      # Sum of all e^totals
      S = np.sum(t_exp)

      # Gradients of out[i] against totals
      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      # Gradients of totals against weights/biases/input
      d_t_d_w = self.last_input
      d_t_d_b = 1
      d_t_d_inputs = self.weights

      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t

      # Gradients of loss against weights/biases/input
      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t

      # Update weights / biases
      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b

      return d_L_d_inputs.reshape(self.last_input_shape)
