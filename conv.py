import numpy as np

'''
二次元のnumpy配列を使用する
今回使用する画像がグレースケールのためRGBを考慮しなくて済む
'''

class Conv3x3:
  # 畳み込みは3*3フィルターを使用する

  def __init__(self, num_filters):
    self.num_filters = num_filters

    #フィルター行列の各値をデフォルトで設定されていた９で割る
    #numpyで作成された各値をそのまま使用すると不具合の原因に繋がる
    self.filters = np.random.randn(num_filters, 3, 3) / 9

  def iterate_regions(self, image):
    '''
    各フィルターが入力画像をどのようにスライドしながら特徴を読み込むのか指定する
    '''
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def forward(self, input):
    '''
    出力行列を算出する
    '''
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    損失計算
    バックプロバケーション
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

    # 損失度合いからフィルターの更新
    self.filters -= learn_rate * d_L_d_filters


    return None
