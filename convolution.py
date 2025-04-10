import numpy as np


class ConvolutionFirstLayer:
    def __init__(self, filters_num):
        self.filters_num = filters_num
        self.filters = np.random.randn(filters_num, 5, 5) / 25

    def forward(self, img):
        self.last_input = img
        h, w = img.shape
        output = np.zeros((h-4, h-4, self.filters_num))
        for im_region, i, j in self.iterate_regions(img):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def iterate_regions(self, img):
        h, w = img.shape

        for i in range(h - 4):
            for j in range(w - 4):
                im_region = img[i:(i + 5), j:(j + 5)]
                yield im_region, i, j

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.filters_num):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters -= learn_rate * d_L_d_filters
        return None


class ConvolutionInerLayer:
    def __init__(self, filters_num, c_in):
        self.filters_num = filters_num
        self.filters = np.random.randn(
            self.filters_num, 5, 5, c_in) / (5 * 5 * c_in)
        self.last_input = None

    def forward(self, img):
        self.last_input = img
        h, w, c_in = img.shape
        output = np.zeros((h - 4, w - 4, self.filters_num))

        for im_region, i, j in self.iterate_regions(img):
            for f in range(self.filters_num):
                output[i, j, f] = np.sum(
                    im_region * self.filters[f], axis=(0, 1, 2))
        return output

    def iterate_regions(self, img):
        h, w, _ = img.shape
        for i in range(h - 4):
            for j in range(w - 4):
                im_region = img[i:(i + 5), j:(j + 5)]
                yield im_region, i, j

    def backprop(self, d_L_d_out, learn_rate):
        h, w, c_in = self.last_input.shape
        d_L_d_filters = np.zeros_like(self.filters)
        d_L_d_inputs = np.zeros_like(self.last_input)
        p_d_L_d_out = np.pad(
            d_L_d_out, [(4, 4), (4, 4), (0, 0)], mode='constant')
        rotated_filters = np.flip(self.filters, (1, 2))

        for im_region, i, j in self.iterate_regions(p_d_L_d_out):
            for c in range(c_in):
                for f in range(self.filters_num):
                    d_L_d_inputs[i, j, c] += np.sum(
                        im_region[:, :, f] * rotated_filters[f, :, :, c])

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.filters_num):
                d_L_d_filters[f] += im_region * d_L_d_out[i, j, f]

        self.filters -= learn_rate * d_L_d_filters
        return d_L_d_inputs
